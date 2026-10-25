"""Z1 timelapse status parsing and the camera-button recording details."""

from types import SimpleNamespace

from carveracontroller.CNC import CNC
from carveracontroller.Controller import Controller
from carveracontroller.main import Makera
from carveracontroller.timelapse import format_timelapse_status, timelapse_is_recording


def _parse_status_line(line):
    controller = Controller(CNC(), lambda _line: None, False)
    controller.parseBracketAngle(line)


def test_timelapse_status_fields():
    _parse_status_line("<Run|MPos:0,0,0|WPos:0,0,0|E:1,1,1,1234,5678|OTA:2,40>")
    assert CNC.vars["state"] == "Run"
    assert CNC.vars["tl_status"] == 1
    assert CNC.vars["tl_transfer"] == 1
    assert CNC.vars["tl_requested"] == 1
    assert CNC.vars["tl_recording"] == 1
    assert CNC.vars["tl_sd_used"] == 1234
    assert CNC.vars["tl_sd_total"] == 5678
    assert CNC.vars["ota_phase"] == 2
    assert CNC.vars["ota_progress"] == 40
    assert timelapse_is_recording(CNC.vars["tl_recording"]) is True


def test_status_without_timelapse_extension_clears_fields():
    CNC.vars["tl_status"] = 1
    CNC.vars["tl_requested"] = 1
    CNC.vars["tl_recording"] = 1
    CNC.vars["ota_phase"] = 4
    _parse_status_line("<Idle|MPos:0,0,0|WPos:0,0,0>")
    assert CNC.vars["tl_status"] == 0
    assert CNC.vars["tl_requested"] == 0
    assert CNC.vars["tl_recording"] == 0
    assert CNC.vars["ota_phase"] == 0
    assert CNC.vars["ota_progress"] == 0


def test_short_timelapse_suffix_leaves_position_and_clears_recording():
    CNC.vars["tl_status"] = 1
    CNC.vars["tl_recording"] = 1
    _parse_status_line("<Run|MPos:1.5,2.5,3.5|WPos:0,0,0|E:1,1>")
    assert CNC.vars["mx"] == 1.5
    assert CNC.vars["state"] == "Run"
    assert CNC.vars["tl_status"] == 0
    assert CNC.vars["tl_recording"] == 0


def test_recording_mark_follows_the_firmware_bit():
    assert timelapse_is_recording(1) is True
    assert timelapse_is_recording(0) is False
    app = SimpleNamespace(
        state="Hold",
        timelapse_recording=False,
        timelapse_status=False,
        timelapse_status_text="",
    )
    CNC.vars["tl_status"] = 1
    CNC.vars["tl_requested"] = 1
    CNC.vars["tl_recording"] = 1
    CNC.vars["tl_transfer"] = 0
    CNC.vars["tl_sd_used"] = 10
    CNC.vars["tl_sd_total"] = 20
    Makera._refresh_timelapse_indicator(None, app)
    assert app.timelapse_recording is True
    assert app.timelapse_status is True
    assert "Recording: yes" in app.timelapse_status_text

    CNC.vars["tl_recording"] = 0
    app.state = "Run"
    Makera._refresh_timelapse_indicator(None, app)
    assert app.timelapse_recording is False
    assert "Recording: no" in app.timelapse_status_text


def test_timelapse_status_reports_e_fields():
    text = format_timelapse_status(
        transfer=0,
        requested=1,
        recording=1,
        sd_used=1200,
        sd_total=8192,
    )
    assert text == "\n".join(
        [
            "Auto Timelapse: on",
            "Recording: yes",
            "File transfer: idle",
            "SD card space: 1200 / 8192 MiB",
        ]
    )
