"""
Network transport for the WHB04 pendant.

The pendant's USB dongle is plugged into a "WHB04 Network Bridge" (an
ESP32-S3 running the firmware from the WHB04-Network-Bridge project), which
exposes the raw HID reports over a TCP connection. This module provides a
drop-in replacement for the hidapi device used by ``whb04.Daemon`` so that all
of the existing report decoding, jog handling and LCD updates are reused
unchanged. Because it only needs the standard library, it also works on
platforms without hidapi (such as Android).

Wire protocol (version 1). The controller is the TCP client; the bridge is the
server (default port 5104). Every message is a frame::

    +--------+--------+-----------------+
    | type   | length | payload         |
    | 1 byte | 1 byte | ``length`` bytes|
    +--------+--------+-----------------+

Bridge -> controller:

- ``0x01`` HELLO: ``[protocol_version, *firmware_version_ascii]``. Sent first.
- ``0x02`` STATUS: ``[dongle_attached]`` (0 or 1). Sent after HELLO and on change.
- ``0x03`` INPUT_REPORT: one raw HID input report, including the report ID.
- ``0x04`` HEARTBEAT: empty. Sent at least every 500 ms.

Controller -> bridge:

- ``0x81`` FEATURE_REPORT: one raw HID feature report, starting with the
  report ID (``0x06`` for the WHB04 LCD).
- ``0x82`` HEARTBEAT: empty.

Either side treats silence for longer than its timeout as a dead connection.
"""

from __future__ import annotations

import logging
import select
import socket
import time
from collections import deque
from collections.abc import Callable

from . import whb04

logger = logging.getLogger(__name__)

DEFAULT_PORT = 5104
PROTOCOL_VERSION = 1

FRAME_HELLO = 0x01
FRAME_STATUS = 0x02
FRAME_INPUT_REPORT = 0x03
FRAME_HEARTBEAT = 0x04
FRAME_FEATURE_REPORT = 0x81
FRAME_CLIENT_HEARTBEAT = 0x82


class BridgeError(Exception):
    """Raised when the bridge connection is unusable (closed, silent, or the dongle went away)."""


def parse_host_port(value: str, default_port: int = DEFAULT_PORT) -> tuple[str, int]:
    """
    Parses ``host``, ``host:port``, ``[ipv6]`` or ``[ipv6]:port``.

    A bare IPv6 address without brackets is accepted as a host with the
    default port. Raises ValueError if the value is empty or the port invalid.
    """
    value = (value or "").strip()
    if not value:
        raise ValueError("No pendant bridge address configured")

    host, port_str = value, ""
    if value.startswith("["):
        end = value.find("]")
        if end < 0:
            raise ValueError(f"Invalid address: {value}")
        host = value[1:end]
        rest = value[end + 1 :]
        if rest.startswith(":"):
            port_str = rest[1:]
        elif rest:
            raise ValueError(f"Invalid address: {value}")
    elif value.count(":") == 1:
        host, port_str = value.split(":", 1)

    host = host.strip()
    if not host:
        raise ValueError(f"Invalid address: {value}")

    if port_str.strip():
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port in address: {value}") from None
        if not 0 < port < 65536:
            raise ValueError(f"Invalid port in address: {value}")
    else:
        port = default_port

    return host, port


class NetworkPendantDevice:
    """
    A connected bridge, presenting the same ``read``/``send_feature_report``
    interface as the hidapi device used by ``whb04.Daemon``.
    """

    HEARTBEAT_INTERVAL = 0.5  # seconds between client heartbeats when otherwise idle
    PEER_TIMEOUT = 1.5  # seconds of bridge silence before the link is considered dead
    DISPLAY_RESEND_INTERVAL = 1.0  # resend an unchanged LCD frame at least this often

    def __init__(self, sock: socket.socket, clock: Callable[[], float] = time.monotonic) -> None:
        self._sock = sock
        self._sock.setblocking(False)
        self._clock = clock
        self._rx_buffer = bytearray()
        self._input_reports: deque[bytes] = deque()
        now = self._clock()
        self._last_rx = now
        self._last_tx = now
        self._closed = False
        self._status_received = False

        self.protocol_version: int | None = None
        self.firmware_version = ""
        self.dongle_attached = False

        # LCD updates arrive from whb04.Daemon as sets of three 8-byte feature
        # reports, rewritten on every loop iteration. Only forward a set when
        # it changes (or periodically) to keep WiFi traffic and USB control
        # transfers on the bridge low.
        self._pending_display: list[bytes] = []
        self._last_display: list[bytes] | None = None
        self._last_display_time = 0.0

    # -- hidapi-compatible interface used by whb04.Daemon ----------------------

    def read(self, size: int, timeout: int | None = None) -> bytes:
        """
        Returns the next input report, or ``b""`` if none arrived within
        ``timeout`` milliseconds. Raises BridgeError if the link died or the
        dongle was unplugged from the bridge.
        """
        deadline = self._clock() + (timeout or 0) / 1000.0
        while True:
            if self._input_reports:
                return self._input_reports.popleft()[:size]
            remaining = deadline - self._clock()
            self._poll(max(0.0, remaining))
            if not self.dongle_attached:
                raise BridgeError("Pendant dongle disconnected from bridge")
            if self._input_reports:
                return self._input_reports.popleft()[:size]
            if self._clock() >= deadline:
                return b""

    def send_feature_report(self, data: bytes) -> int:
        data = bytes(data)
        if len(data) > 255:
            raise ValueError("Feature report too long")

        is_display_packet = len(data) == 8 and data[0] == 0x06
        if not is_display_packet:
            self._send_frame(FRAME_FEATURE_REPORT, data)
            return len(data)

        # The first packet of an LCD update carries the 0xFDFE magic.
        if data[1] == 0xFE and data[2] == 0xFD:
            self._pending_display = [data]
        elif self._pending_display and len(self._pending_display) < 3:
            self._pending_display.append(data)
        else:
            # Out-of-sequence packet; forward as-is rather than guess.
            self._send_frame(FRAME_FEATURE_REPORT, data)
            return len(data)

        if len(self._pending_display) == 3:
            now = self._clock()
            if (
                self._pending_display != self._last_display
                or now - self._last_display_time >= self.DISPLAY_RESEND_INTERVAL
            ):
                payload = b"".join(
                    bytes((FRAME_FEATURE_REPORT, len(packet))) + packet for packet in self._pending_display
                )
                self._send_raw(payload)
                self._last_display = self._pending_display
                self._last_display_time = now
            self._pending_display = []
        return len(data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> NetworkPendantDevice:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -- connection management -------------------------------------------------

    def handshake(self, timeout: float) -> None:
        """Waits for the bridge's HELLO and initial STATUS frames."""
        deadline = self._clock() + timeout
        while self.protocol_version is None or not self._status_received:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise BridgeError("Timed out waiting for bridge handshake")
            self._poll(remaining)
        if self.protocol_version != PROTOCOL_VERSION:
            raise BridgeError(
                f"Unsupported bridge protocol version {self.protocol_version} (expected {PROTOCOL_VERSION})"
            )

    def poll(self, timeout: float) -> None:
        """Processes incoming frames for up to ``timeout`` seconds."""
        self._poll(timeout)

    # -- internals -------------------------------------------------------------

    def _poll(self, timeout: float) -> None:
        if self._closed:
            raise BridgeError("Connection closed")

        now = self._clock()
        if now - self._last_tx >= self.HEARTBEAT_INTERVAL:
            self._send_frame(FRAME_CLIENT_HEARTBEAT, b"")

        # Never sleep past the point where we would owe the bridge a heartbeat
        # or would declare it dead.
        timeout = min(
            timeout,
            max(0.0, self._last_tx + self.HEARTBEAT_INTERVAL - now),
            max(0.0, self._last_rx + self.PEER_TIMEOUT - now),
        )

        readable, _, errored = select.select([self._sock], [], [self._sock], timeout)
        if errored:
            raise BridgeError("Socket error")
        if readable:
            try:
                chunk = self._sock.recv(4096)
            except (BlockingIOError, InterruptedError):
                chunk = None
            except OSError as e:
                raise BridgeError(f"Connection lost: {e}") from e
            if chunk is not None:
                if not chunk:
                    raise BridgeError("Bridge closed the connection")
                self._last_rx = self._clock()
                self._rx_buffer.extend(chunk)
                self._parse_frames()

        if self._clock() - self._last_rx > self.PEER_TIMEOUT:
            raise BridgeError("Bridge stopped responding")

    def _parse_frames(self) -> None:
        buf = self._rx_buffer
        while len(buf) >= 2:
            frame_type, length = buf[0], buf[1]
            if len(buf) < 2 + length:
                break
            payload = bytes(buf[2 : 2 + length])
            del buf[: 2 + length]
            self._handle_frame(frame_type, payload)

    def _handle_frame(self, frame_type: int, payload: bytes) -> None:
        if frame_type == FRAME_INPUT_REPORT:
            if payload:
                self._input_reports.append(payload)
        elif frame_type == FRAME_HEARTBEAT:
            pass
        elif frame_type == FRAME_STATUS:
            self._status_received = True
            attached = bool(payload and payload[0])
            if attached != self.dongle_attached:
                logger.info("Bridge reports pendant dongle %s", "attached" if attached else "detached")
            self.dongle_attached = attached
            if not attached:
                # Reports queued before the detach are stale.
                self._input_reports.clear()
        elif frame_type == FRAME_HELLO:
            if not payload:
                raise BridgeError("Malformed HELLO from bridge")
            self.protocol_version = payload[0]
            self.firmware_version = payload[1:].decode("ascii", errors="replace")
        else:
            logger.debug("Ignoring unknown bridge frame type 0x%02X", frame_type)

    def _send_frame(self, frame_type: int, payload: bytes) -> None:
        self._send_raw(bytes((frame_type, len(payload))) + payload)

    def _send_raw(self, data: bytes) -> None:
        if self._closed:
            raise BridgeError("Connection closed")
        view = memoryview(data)
        deadline = self._clock() + self.PEER_TIMEOUT
        while view:
            try:
                sent = self._sock.send(view)
            except (BlockingIOError, InterruptedError):
                if self._clock() > deadline:
                    raise BridgeError("Timed out sending to bridge") from None
                select.select([], [self._sock], [], 0.05)
                continue
            except OSError as e:
                raise BridgeError(f"Connection lost: {e}") from e
            view = view[sent:]
        self._last_tx = self._clock()


class NetworkDaemon(whb04.Daemon):
    """
    A ``whb04.Daemon`` that talks to a WHB04 Network Bridge instead of a
    locally attached USB dongle. ``on_connect`` fires once the bridge is
    reachable *and* reports the pendant dongle attached.
    """

    CONNECT_TIMEOUT = 3.0  # seconds for TCP connect + handshake
    RETRY_INTERVAL = 2.0  # seconds between connection attempts

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        callback_executor: Callable[[Callable[[], None]], None] = lambda f: f(),
    ) -> None:
        super().__init__(callback_executor)
        self.host = host
        self.port = port
        self._last_error: str | None = None

    def stop(self) -> None:
        if not self._is_running:
            return
        self._is_running = False
        # The worker checks _is_running at least every 200 ms, except while
        # blocked in a TCP connect (up to CONNECT_TIMEOUT). Don't stall the UI
        # thread for that; the thread is a daemon and exits on its own, and
        # never hands out a device once stopped.
        self._daemon_thread.join(timeout=0.5)

    # Duck-typed stand-in for the hidapi device the base class annotates.
    def _connect(self, poll_interval: float = 0.1) -> NetworkPendantDevice | None:  # type: ignore[override]
        while self._is_running:
            device = None
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.CONNECT_TIMEOUT)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                device = NetworkPendantDevice(sock)
                device.handshake(self.CONNECT_TIMEOUT)
                self._log_state(f"connected to bridge {self.host}:{self.port} (firmware {device.firmware_version})")

                while self._is_running and not device.dongle_attached:
                    device.poll(0.2)

                if self._is_running:
                    return device
                device.close()
                return None
            except (OSError, BridgeError, ValueError) as e:
                if device is not None:
                    device.close()
                self._log_state(f"cannot use bridge {self.host}:{self.port}: {e}", warning=True)

            deadline = time.monotonic() + self.RETRY_INTERVAL
            while self._is_running and time.monotonic() < deadline:
                time.sleep(poll_interval)

        return None

    def _log_state(self, message: str, warning: bool = False) -> None:
        # Avoid spamming the log with the same failure every retry.
        if message == self._last_error:
            return
        self._last_error = message
        (logger.warning if warning else logger.info)("WHB04 network pendant: %s", message)
