"""Z1 timelapse fields carried on the machine status report.

The firmware appends this suffix before the closing ``>``:

``|E:<transfer>,<record-requested>,<recording>,<sd-used>,<sd-total>|OTA:<phase>,<progress>``

The red recording mark follows the firmware ``recording`` field. The camera
settings panel shows the ``|E:`` fields while it is open.
"""

from __future__ import annotations

from typing import Callable

Translate = Callable[[str], str]


def timelapse_is_recording(recording: int) -> bool:
    """True when the firmware reports that it is writing a timelapse."""
    return bool(recording)


def format_timelapse_status(
    *,
    transfer: int,
    requested: int,
    recording: int,
    sd_used: int,
    sd_total: int,
    translate: Translate = lambda text: text,
) -> str:
    """Recording details from the ``|E:`` status fields, for the camera settings panel."""
    t = translate
    lines = [
        t("Auto Timelapse: on") if requested else t("Auto Timelapse: off"),
        t("Recording: yes") if recording else t("Recording: no"),
        t("File transfer: active") if transfer else t("File transfer: idle"),
        t("SD card space: %s / %s MiB") % (sd_used, sd_total),
    ]
    return "\n".join(lines)
