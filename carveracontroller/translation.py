# This module provides translation functionality for the Carvera Controller
# application that can be shared across all modules in the application.

from __future__ import annotations

import gettext
import locale
import os

from kivy.lang import Observable
from kivy.utils import platform as kivy_platform

LANGS = {
    "en": "English",
    "zh-CN": "中文简体(Simplified Chinese)",
}

# Regions and scripts that use Traditional Chinese. The only shipped Chinese
# catalog is Simplified, so these fall back to English.
_TRADITIONAL_CHINESE = {"hant", "tw", "hk", "mo"}


class Lang(Observable):
    observers = []
    lang = None

    def __init__(self, defaultlang):
        super().__init__()
        self.ugettext = None
        self.lang = defaultlang
        self.switch_lang(self.lang)

    def _(self, text):
        return self.ugettext(text)

    def fbind(self, name, func, args, **kwargs):
        if name == "_":
            self.observers.append((func, args, kwargs))
        else:
            return super().fbind(name, func, *args, **kwargs)

    def funbind(self, name, func, args, **kwargs):
        if name == "_":
            key = (func, args, kwargs)
            if key in self.observers:
                self.observers.remove(key)
        else:
            return super().funbind(name, func, *args, **kwargs)

    def switch_lang(self, lang):
        # get the right locales directory, and instanciate a gettext
        locale_dir = os.path.join(os.path.dirname(__file__), "locales")
        locales = None
        try:
            locales = gettext.translation(lang, locale_dir, languages=[lang])
        except:
            pass
        if locales == None:
            locales = gettext.NullTranslations()
        self.ugettext = locales.gettext
        self.lang = lang

        # update all the kv rules attached to this text
        for func, largs, kwargs in self.observers:
            func(largs, None, None)


# Proxy class is needed to allow for from carveracontroller.translation import tr.
# Without proxy, the initialization of the translation module would fail
# because the tr object is copied from the module to the caller's namespace
# before the translation is initialized.
class TrProxy:
    def __getattr__(self, name):
        if _translator is None:
            raise RuntimeError("Translation not initialized")
        return getattr(_translator, name)


_translator: Lang | None = Lang("en")
tr = TrProxy()


def supported_language_for_tag(tag: str | None) -> str | None:
    """Map a locale or BCP 47 tag onto a shipped language key."""
    if not tag:
        return None
    parts = tag.replace("_", "-").lower().split("-")
    language = parts[0]
    if language == "zh":
        if {*parts[1:]} & _TRADITIONAL_CHINESE:
            return None
        return "zh-CN" if "zh-CN" in LANGS else None
    for lang_key in LANGS:
        if lang_key.split("-", 1)[0] == language:
            return lang_key
    return None


def _android_language_tag() -> str | None:
    """Device locale via the JVM default, which Android sets to the system language."""
    try:
        from jnius import autoclass
    except ImportError:
        return None
    try:
        return str(autoclass("java.util.Locale").getDefault().toLanguageTag())
    except Exception:
        return None


def _posix_language_tag() -> str | None:
    try:
        lang, _ = locale.getlocale()
    except Exception:
        return None
    return lang


def default_language() -> str:
    if kivy_platform == "android":
        return supported_language_for_tag(_android_language_tag()) or "en"
    return supported_language_for_tag(_posix_language_tag()) or "en"


def init(langname: str | None = None):
    if langname not in LANGS:
        langname = default_language()

    global _translator
    _translator = Lang(langname)
