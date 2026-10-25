"""Bed editor field helpers (no Kivy widget tree)."""

import pytest

from carveracontroller.addons.beds.catalog import CUSTOM_CATALOG_ID
from carveracontroller.addons.beds.ui.BedEditorPopup import (
    BedEditorPopup,
    _optional_mm_text,
    _parse_finite_float,
)


def test_optional_mm_text_blank_until_user_types():
    assert _optional_mm_text(None) == ""


def test_optional_mm_text_keeps_zero_and_negatives():
    assert _optional_mm_text(0) == "0"
    assert _optional_mm_text(0.0) == "0"
    assert _optional_mm_text(-111.5) == "-111.5"


def test_machine_z_empty_is_required():
    with pytest.raises(ValueError):
        _parse_finite_float("  ", "Machine Z")


def test_machine_z_accepts_typed_zero():
    assert _parse_finite_float("0", "Machine Z") == 0.0
    assert _parse_finite_float("-108.3", "Machine Z") == -108.3


class _Text:
    def __init__(self, text=""):
        self.text = text


def test_selecting_z1_smw_writes_x_offset_into_machine_fields(monkeypatch):
    from carveracontroller.CNC import CNC

    monkeypatch.setitem(CNC.vars, "anchor1_x", -360.158)
    monkeypatch.setitem(CNC.vars, "anchor1_y", -234.568)
    monkeypatch.setitem(CNC.vars, "anchor_width", 15.0)

    popup = BedEditorPopup.__new__(BedEditorPopup)
    popup._suppress_model_change = False
    popup._model_pairs = [
        ("MDF", "Z1_MDF"),
        ("SMW (Metric)", "Z1_SMW_Metric"),
        ("Custom…", CUSTOM_CATALOG_ID),
    ]
    popup._material_pairs = [("MDF", "mdf"), ("Aluminum", "aluminum")]
    mcs_x = _Text("0")
    mcs_y = _Text("0")
    popup.ids = {
        "spn_model": _Text("SMW (Metric)"),
        "spn_material": _Text(""),
        "ti_name": _Text("kept"),
        "txt_mcs_x": mcs_x,
        "txt_mcs_y": mcs_y,
    }
    popup._sync_custom_ui = lambda: None
    popup._refresh_thickness_hint = lambda: None

    popup.on_model_selected()

    assert float(mcs_x.text) == pytest.approx(-375.158 - 0.33 * 25.4)
    assert float(mcs_y.text) == pytest.approx(-249.568)
    assert popup.ids.ti_name.text == "kept"

    popup.ids.spn_model.text = "MDF"
    popup.on_model_selected()
    assert float(mcs_x.text) == pytest.approx(-375.158)
    assert float(mcs_y.text) == pytest.approx(-249.568)
