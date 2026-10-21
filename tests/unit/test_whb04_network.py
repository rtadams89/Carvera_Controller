"""Tests for the WHB04 network pendant transport, using a simulated bridge."""

from __future__ import annotations

import socket
import struct
import threading
import time

import pytest

from carveracontroller.addons.pendant import whb04, whb04_network
from carveracontroller.addons.pendant.whb04_network import (
    FRAME_CLIENT_HEARTBEAT,
    FRAME_FEATURE_REPORT,
    FRAME_HEARTBEAT,
    FRAME_HELLO,
    FRAME_INPUT_REPORT,
    FRAME_STATUS,
    BridgeError,
    NetworkDaemon,
    NetworkPendantDevice,
    parse_host_port,
)


def frame(frame_type: int, payload: bytes = b"") -> bytes:
    return bytes((frame_type, len(payload))) + payload


def input_report(button1: int = 0, axis: whb04.Axis = whb04.Axis.X, jog: int = 0) -> bytes:
    return struct.pack("BBBBBBbB", 0x04, 0, button1, 0, whb04.StepSize.STEP_0_1.value, axis.value, jog, 0)


def read_frames(conn: socket.socket, buffer: bytearray) -> list[tuple[int, bytes]]:
    frames = []
    while len(buffer) >= 2 and len(buffer) >= 2 + buffer[1]:
        length = buffer[1]
        frames.append((buffer[0], bytes(buffer[2 : 2 + length])))
        del buffer[: 2 + length]
    return frames


class FakeBridge:
    """Minimal single-client bridge server for tests."""

    def __init__(self, dongle_attached: bool = True, heartbeat: bool = True, protocol_version: int = 1) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.dongle_attached = dongle_attached
        self.heartbeat = heartbeat
        self.protocol_version = protocol_version
        self.conn: socket.socket | None = None
        self.connected = threading.Event()
        self.received: list[tuple[int, bytes]] = []
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self.server.settimeout(0.1)
        while self._running:
            try:
                conn, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self.conn = conn
            self.send(frame(FRAME_HELLO, bytes((self.protocol_version,)) + b"test-1.0"))
            self.send(frame(FRAME_STATUS, bytes((int(self.dongle_attached),))))
            self.connected.set()
            buffer = bytearray()
            conn.settimeout(0.1)
            last_hb = time.monotonic()
            while self._running:
                if self.heartbeat and time.monotonic() - last_hb > 0.3:
                    self.send(frame(FRAME_HEARTBEAT))
                    last_hb = time.monotonic()
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                buffer.extend(data)
                with self._lock:
                    self.received.extend(read_frames(conn, buffer))
            conn.close()
            self.conn = None
            self.connected.clear()

    def send(self, data: bytes) -> None:
        if self.conn is not None:
            try:
                self.conn.sendall(data)
            except OSError:
                pass

    def received_of_type(self, frame_type: int) -> list[bytes]:
        with self._lock:
            return [payload for t, payload in self.received if t == frame_type]

    def close(self) -> None:
        self._running = False
        self.server.close()
        self._thread.join(timeout=2)


@pytest.fixture
def bridge():
    b = FakeBridge()
    yield b
    b.close()


def wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("192.168.1.50", ("192.168.1.50", 5104)),
        (" 192.168.1.50:6000 ", ("192.168.1.50", 6000)),
        ("whb04-bridge.lan", ("whb04-bridge.lan", 5104)),
        ("[fe80::1]:7000", ("fe80::1", 7000)),
        ("[fe80::1]", ("fe80::1", 5104)),
        ("fe80::1", ("fe80::1", 5104)),
    ],
)
def test_parse_host_port(value, expected):
    assert parse_host_port(value) == expected


@pytest.mark.parametrize("value", ["", "   ", "host:0", "host:70000", "host:abc", ":5104", "[fe80::1"])
def test_parse_host_port_rejects_invalid(value):
    with pytest.raises(ValueError):
        parse_host_port(value)


def test_daemon_decodes_reports_and_sends_display(bridge):
    daemon = NetworkDaemon("127.0.0.1", bridge.port)
    events = []
    connected = threading.Event()
    daemon.on_connect = lambda _d: connected.set()
    daemon.on_button_press = lambda _d, button: events.append(("press", button))
    daemon.on_axis_change = lambda _d, axis: events.append(("axis", axis))
    daemon.start()
    try:
        assert connected.wait(3)
        bridge.send(frame(FRAME_INPUT_REPORT, input_report(button1=whb04.Button.STOP.value, axis=whb04.Axis.Y)))
        assert wait_for(lambda: ("press", whb04.Button.STOP) in events)
        assert ("axis", whb04.Axis.Y) in events

        # The daemon refreshes the LCD continuously; the bridge should get
        # complete three-packet sets of 0x06 feature reports.
        assert wait_for(lambda: len(bridge.received_of_type(FRAME_FEATURE_REPORT)) >= 3)
        reports = bridge.received_of_type(FRAME_FEATURE_REPORT)
        assert all(len(r) == 8 and r[0] == 0x06 for r in reports)
        assert reports[0][1:3] == b"\xfe\xfd"
    finally:
        daemon.stop()


def test_unchanged_display_is_not_resent_every_loop(bridge):
    daemon = NetworkDaemon("127.0.0.1", bridge.port)
    connected = threading.Event()
    daemon.on_connect = lambda _d: connected.set()
    daemon.start()
    try:
        assert connected.wait(3)
        time.sleep(0.6)
        # The daemon loop runs ~10x/s; with dedupe only the initial set
        # (and no periodic resend yet) should have been sent.
        assert len(bridge.received_of_type(FRAME_FEATURE_REPORT)) == 3

        daemon.set_display_position(whb04.Axis.X, 12.5)
        assert wait_for(lambda: len(bridge.received_of_type(FRAME_FEATURE_REPORT)) == 6)
    finally:
        daemon.stop()


def test_dongle_detach_reports_disconnect_and_reconnects(bridge):
    daemon = NetworkDaemon("127.0.0.1", bridge.port)
    daemon.RETRY_INTERVAL = 0.1
    connects, disconnects = [], []
    daemon.on_connect = lambda _d: connects.append(time.monotonic())
    daemon.on_disconnect = lambda _d: disconnects.append(time.monotonic())
    daemon.start()
    try:
        assert wait_for(lambda: len(connects) == 1)
        bridge.dongle_attached = False
        bridge.send(frame(FRAME_STATUS, b"\x00"))
        assert wait_for(lambda: len(disconnects) == 1)

        # Dongle comes back: the bridge announces it on the (re)connection.
        bridge.dongle_attached = True
        bridge.send(frame(FRAME_STATUS, b"\x01"))
        assert wait_for(lambda: len(connects) == 2)
    finally:
        daemon.stop()


def test_not_connected_while_dongle_absent():
    b = FakeBridge(dongle_attached=False)
    daemon = NetworkDaemon("127.0.0.1", b.port)
    connected = threading.Event()
    daemon.on_connect = lambda _d: connected.set()
    daemon.start()
    try:
        assert b.connected.wait(3)
        assert not connected.wait(0.5)
        b.send(frame(FRAME_STATUS, b"\x01"))
        assert connected.wait(3)
    finally:
        daemon.stop()
        b.close()


def test_silent_bridge_is_treated_as_disconnected():
    b = FakeBridge(heartbeat=False)
    daemon = NetworkDaemon("127.0.0.1", b.port)
    connected, disconnected = threading.Event(), threading.Event()
    daemon.on_connect = lambda _d: connected.set()
    daemon.on_disconnect = lambda _d: disconnected.set()
    daemon.start()
    try:
        assert connected.wait(3)
        assert disconnected.wait(NetworkPendantDevice.PEER_TIMEOUT + 1.5)
    finally:
        daemon.stop()
        b.close()


def test_client_sends_heartbeats(bridge):
    sock = socket.create_connection(("127.0.0.1", bridge.port))
    device = NetworkPendantDevice(sock)
    try:
        device.handshake(2)
        assert device.firmware_version == "test-1.0"
        assert device.dongle_attached
        for _ in range(8):
            device.read(8, timeout=100)
        assert bridge.received_of_type(FRAME_CLIENT_HEARTBEAT)
    finally:
        device.close()


def test_protocol_version_mismatch_is_rejected():
    b = FakeBridge(protocol_version=99)
    sock = socket.create_connection(("127.0.0.1", b.port))
    device = NetworkPendantDevice(sock)
    try:
        with pytest.raises(BridgeError):
            device.handshake(2)
    finally:
        device.close()
        b.close()


def test_stop_without_bridge_returns_quickly():
    # Nothing listening on this port: connect fails, daemon retries.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    daemon = NetworkDaemon("127.0.0.1", port)
    daemon.start()
    time.sleep(0.2)
    started = time.monotonic()
    daemon.stop()
    assert time.monotonic() - started < 1.0


def test_module_works_without_hidapi():
    # The network transport must not depend on hidapi (absent on Android).
    assert hasattr(whb04_network, "NetworkDaemon")
    assert issubclass(NetworkDaemon, whb04.Daemon)


def _make_network_pendant(monkeypatch, address: str, **controller_overrides):
    from types import SimpleNamespace

    from carveracontroller.addons.pendant import pendant as pendant_module

    monkeypatch.setattr(
        pendant_module.Config,
        "get",
        lambda section, key, fallback=None: address if key == "pendant_network_address" else fallback,
    )
    controller = SimpleNamespace(continuous_jog_active=False, stopped=[], **controller_overrides)
    controller.stopContinuousJog = lambda: controller.stopped.append(True)
    reports = []
    pendant = pendant_module.WHB04Network(
        controller,
        SimpleNamespace(vars={}),
        None,
        None,
        lambda: True,
        lambda: None,
        lambda: None,
        lambda: None,
        lambda: reports.append("connected"),
        lambda: reports.append("disconnected"),
    )
    return pendant, controller, reports


def test_network_pendant_without_address_stays_idle(monkeypatch):
    from carveracontroller.addons.pendant import pendant as pendant_module

    pendant, _, _ = _make_network_pendant(monkeypatch, "")
    assert isinstance(pendant._daemon, pendant_module._IdleDaemon)
    pendant.close()


def test_network_pendant_uses_configured_address(monkeypatch):
    pendant, _, _ = _make_network_pendant(monkeypatch, "127.0.0.1:1")
    try:
        assert isinstance(pendant._daemon, NetworkDaemon)
        assert (pendant._daemon.host, pendant._daemon.port) == ("127.0.0.1", 1)
    finally:
        pendant.close()
    assert not pendant._daemon._is_running


def test_network_pendant_disconnect_stops_continuous_jog(monkeypatch):
    pendant, controller, reports = _make_network_pendant(monkeypatch, "127.0.0.1:1")
    try:
        controller.continuous_jog_active = True
        pendant._handle_disconnect(pendant._daemon)
        assert controller.stopped == [True]
        assert reports == ["disconnected"]
    finally:
        pendant.close()
