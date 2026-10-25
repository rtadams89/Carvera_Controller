"""Default UI language selection."""

import pytest

from carveracontroller import translation


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        (None, None),
        ("", None),
        ("en", "en"),
        ("en_AU", "en"),
        ("en-US", "en"),
        ("zh", "zh-CN"),
        ("zh_CN", "zh-CN"),
        ("zh-CN", "zh-CN"),
        ("zh-Hans", "zh-CN"),
        ("zh-Hans-CN", "zh-CN"),
        ("zh-SG", "zh-CN"),
        ("zh-TW", None),
        ("zh-HK", None),
        ("zh-Hant", None),
        ("zh-Hant-TW", None),
        ("fr-FR", None),
    ],
)
def test_supported_language_for_tag(tag, expected):
    assert translation.supported_language_for_tag(tag) == expected


def test_posix_c_locale_is_english(monkeypatch):
    monkeypatch.setattr(translation, "kivy_platform", "macosx")
    monkeypatch.setattr(translation, "_posix_language_tag", lambda: "C")

    assert translation.default_language() == "en"


def test_posix_locale_selects_shipped_language(monkeypatch):
    monkeypatch.setattr(translation, "kivy_platform", "linux")
    monkeypatch.setattr(translation, "_posix_language_tag", lambda: "zh_CN")

    assert translation.default_language() == "zh-CN"


def test_android_uses_configuration_locale_not_posix(monkeypatch):
    monkeypatch.setattr(translation, "kivy_platform", "android")
    monkeypatch.setattr(translation, "_android_language_tag", lambda: "en-AU")
    monkeypatch.setattr(translation, "_posix_language_tag", lambda: "C")

    assert translation.default_language() == "en"


def test_android_simplified_chinese(monkeypatch):
    monkeypatch.setattr(translation, "kivy_platform", "android")
    monkeypatch.setattr(translation, "_android_language_tag", lambda: "zh-Hans-CN")

    assert translation.default_language() == "zh-CN"


def test_android_unsupported_language_stays_english(monkeypatch):
    monkeypatch.setattr(translation, "kivy_platform", "android")
    monkeypatch.setattr(translation, "_android_language_tag", lambda: "fr-FR")
    monkeypatch.setattr(translation, "_posix_language_tag", lambda: "zh_CN")

    assert translation.default_language() == "en"
