"""Tooltip layout stays tight when live text changes."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import BooleanProperty, NumericProperty
from kivy.uix.widget import Widget

from carveracontroller.addons.tooltips.Tooltips import (
    TOOLTIP_MIN_WIDTH,
    ToolTipButton,
    ToolTipContentLabel,
    _compute_tooltip_box_size,
    _sync_tooltip_spacing,
)

KV_PATH = Path(__file__).resolve().parents[2] / "carveracontroller" / "addons" / "tooltips" / "Tooltips.kv"
PAD = 20


class _TooltipTestApp(App):
    show_tooltips = BooleanProperty(True)
    tooltip_delay = NumericProperty(0)

    def load_kv(self, filename=None):
        return False

    def build(self):
        return Widget()


@pytest.fixture
def tooltip_kv():
    app = App.get_running_app()
    if app is None:
        app = _TooltipTestApp()
    path = str(KV_PATH)
    if path not in Builder.files:
        Builder.load_file(path)
    return app


@contextmanager
def tooltip_button(**kwargs):
    kwargs.setdefault("text", "")
    kwargs.setdefault("show_tooltips", True)
    btn = ToolTipButton(**kwargs)
    try:
        yield btn
    finally:
        Clock.unschedule(btn.display_tooltip)
        btn.close_tooltip()
        Window.unbind(mouse_pos=btn.on_mouse_pos)


def test_sync_spacing_only_when_both_contents_present():
    tip = SimpleNamespace(spacing=15)
    _sync_tooltip_spacing(tip, has_text=True, has_image=False)
    assert tip.spacing == 0
    _sync_tooltip_spacing(tip, has_text=False, has_image=True)
    assert tip.spacing == 0
    _sync_tooltip_spacing(tip, has_text=True, has_image=True)
    assert tip.spacing == 15
    _sync_tooltip_spacing(None, has_text=True, has_image=True)


def test_text_only_box_size_uses_min_width_and_ignores_spacing():
    width, height = _compute_tooltip_box_size(
        50, 18, 0, 0, has_text=True, has_image=False, horizontal=False, spacing=15
    )
    assert (width, height) == (TOOLTIP_MIN_WIDTH + PAD, 18 + PAD)


def test_vertical_image_and_text_box_size_unchanged():
    width, height = _compute_tooltip_box_size(
        80, 18, 40, 60, has_text=True, has_image=True, horizontal=False, spacing=15
    )
    assert width == max(80 + PAD, 40 + PAD, TOOLTIP_MIN_WIDTH + PAD)
    assert height == 18 + 60 + PAD


def test_horizontal_image_and_text_includes_spacing():
    width, height = _compute_tooltip_box_size(
        80, 18, 40, 32, has_text=True, has_image=True, horizontal=True, spacing=15
    )
    assert width == max(80 + 40 + PAD + 15, TOOLTIP_MIN_WIDTH + PAD)
    assert height == max(18, 32) + PAD


def test_content_label_text_change_stays_one_line():
    label = ToolTipContentLabel(text="Hide simulation")
    label.refresh_text_size()
    first_height = label.texture_size[1]
    label.text = "Show simulation"
    assert label.texture_size[1] == first_height
    assert first_height > 0
    assert label.text_size[1] is None


def test_live_text_change_does_not_add_top_gap(tooltip_kv):
    with tooltip_button(tooltip_txt="Hide simulation") as btn:
        tip = btn._tooltip
        label = tip.ids.tooltip_label
        tip.do_layout()
        height_before = tip.height
        gap_before = tip.top - label.top
        width_before = tip.width

        btn.tooltip_txt = "Show simulation"
        tip.do_layout()

        assert tip.spacing == 0
        assert tip.height == height_before
        assert tip.width == width_before == TOOLTIP_MIN_WIDTH + PAD
        assert tip.top - label.top == gap_before
        assert gap_before == pytest.approx(tip.padding[1], abs=1)


def test_image_tooltips_keep_spacing(tooltip_kv):
    with tooltip_button(tooltip_txt="Probe diameter") as btn:
        tip = btn._tooltip
        tip.ids.tooltip_image.size = (80, 80)
        btn._update_tooltip_size()
        assert tip.spacing == 15

        btn.tooltip_horizontal = True
        tip.ids.tooltip_image.size = (32, 32)
        btn._update_tooltip_size()
        assert tip.spacing == 15
        assert tip.orientation == "horizontal"


def test_markup_text_change_does_not_add_spacing(tooltip_kv):
    with tooltip_button(tooltip_txt="[b]Hide simulation[/b]", tooltip_markup=True) as btn:
        tip = btn._tooltip
        label = tip.ids.tooltip_label
        tip.do_layout()
        height_before = tip.height
        gap_before = tip.top - label.top

        btn.tooltip_txt = "[b]Show simulation[/b]"
        tip.do_layout()

        assert tip.spacing == 0
        assert tip.height == height_before
        assert tip.top - label.top == gap_before
        assert gap_before == pytest.approx(tip.padding[1], abs=1)
