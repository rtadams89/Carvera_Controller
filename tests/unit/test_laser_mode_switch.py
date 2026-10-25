"""Laser Mode switches must track whether laser mode was actually entered."""

from types import SimpleNamespace

from carveracontroller.main import Makera


def _switch(active=False):
    return SimpleNamespace(active=active, set_flag=False)


def _host(switch_active):
    tool_switch = _switch(switch_active)
    host = SimpleNamespace(
        control_list={"laser_mode": [0.0, True]},
        laser_drop_down=SimpleNamespace(opened=False, dismiss=lambda: None, switch=_switch(False)),
        tool_drop_down=SimpleNamespace(ids=SimpleNamespace(switch=tool_switch)),
        diagnose_popup=SimpleNamespace(sw_laser=SimpleNamespace(set_flag=False, switch=_switch(False))),
    )
    host._set_switch_off_silently = lambda *args, **kwargs: Makera._set_switch_off_silently(host, *args, **kwargs)
    return host


def test_disabling_from_tool_switch_does_not_stick_set_flag():
    """The tool switch is already off when its own Off event reaches update_control."""
    host = _host(switch_active=False)

    Makera.update_control(host, "laser_mode", False)

    switch = host.tool_drop_down.ids.switch
    assert switch.active is False
    assert switch.set_flag is False
    assert host.control_list["laser_mode"][1] is False


def test_disabling_from_laser_dropdown_turns_tool_switch_off():
    host = _host(switch_active=True)

    Makera.update_control(host, "laser_mode", False)

    switch = host.tool_drop_down.ids.switch
    assert switch.active is False
    assert switch.set_flag is True


def test_cancel_turns_laser_switches_off():
    host = _host(switch_active=True)
    host.laser_drop_down.switch.active = True
    host.diagnose_popup.sw_laser.switch.active = True

    Makera.cancel_laser_mode_entry(host)

    tool_switch = host.tool_drop_down.ids.switch
    laser_switch = host.laser_drop_down.switch
    diagnose = host.diagnose_popup.sw_laser
    assert tool_switch.active is False
    assert tool_switch.set_flag is True
    assert laser_switch.active is False
    assert laser_switch.set_flag is True
    assert diagnose.switch.active is False
    assert diagnose.set_flag is True


def test_cancel_when_switch_already_off_does_not_stick_set_flag():
    host = _host(switch_active=False)

    Makera.cancel_laser_mode_entry(host)

    switch = host.tool_drop_down.ids.switch
    assert switch.active is False
    assert switch.set_flag is False
