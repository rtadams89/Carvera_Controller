"""Z1 camera reconnect stays alive when the stream raises outside OSError."""

from carveracontroller.addons.camera.Z1Camera import CameraStreamClosed, Z1Camera


def test_unexpected_camera_error_retries_instead_of_sticking_the_panel(monkeypatch):
    created = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            created.append(self)
            if len(created) == 1:
                raise RuntimeError("bad frame")

        def close(self):
            return None

        def send(self, opcode, payload=b""):
            return None

        def read_frame(self):
            camera.stop()
            raise CameraStreamClosed

    monkeypatch.setattr("carveracontroller.addons.camera.Z1Camera.WebSocketClient", FakeClient)
    monkeypatch.setattr("carveracontroller.addons.camera.Z1Camera.RECONNECT_DELAY", 0)
    camera = Z1Camera(on_frame=lambda jpeg: None, on_streaming=lambda streaming: None, on_reconnecting=lambda on: None)

    camera.start("127.0.0.1")
    camera._reader.join(timeout=2)

    assert len(created) == 2
    assert camera._reader.is_alive() is False
    assert camera.is_streaming() is False
