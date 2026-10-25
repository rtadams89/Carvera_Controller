"""Shared window-width breakpoint for popups."""

COMPACT_WIDTH_DP = 720


def is_compact_width(window_width: float, *, threshold: float = COMPACT_WIDTH_DP) -> bool:
    return float(window_width) < float(threshold)
