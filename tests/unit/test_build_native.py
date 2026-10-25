"""Windows MSVC discovery for scripts/build_native.py."""

import importlib.util
import os
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_native.py"
_spec = importlib.util.spec_from_file_location("build_native", _SCRIPT)
assert _spec is not None and _spec.loader is not None
build_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_native)


def test_parse_env_block_keeps_values_that_contain_equals():
    text = "\ufeffPath=C:\\MSVC\\bin;C:\\Windows\r\nINCLUDE=C:\\inc=extra\r\n\r\n=bad\r\n"
    assert build_native._parse_env_block(text) == {
        "Path": r"C:\MSVC\bin;C:\Windows",
        "INCLUDE": r"C:\inc=extra",
    }


def test_find_vcvarsall_prefers_vswhere_find(monkeypatch, tmp_path):
    vswhere = tmp_path / "vswhere.exe"
    bat = tmp_path / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    bat.parent.mkdir(parents=True)
    bat.write_text("", encoding="utf-8")

    monkeypatch.setattr(build_native, "_vswhere_exe", lambda: vswhere)

    def fake_output(exe, extra):
        assert exe == vswhere
        assert extra[0] == "-find"
        return str(bat)

    monkeypatch.setattr(build_native, "_vswhere_output", fake_output)
    assert build_native._find_vcvarsall() == bat


def test_find_vcvarsall_uses_installation_path_when_find_is_empty(monkeypatch, tmp_path):
    install = tmp_path / "VS"
    bat = install / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    bat.parent.mkdir(parents=True)
    bat.write_text("", encoding="utf-8")

    monkeypatch.setattr(build_native, "_vswhere_exe", lambda: tmp_path / "vswhere.exe")

    def fake_output(exe, extra):
        if extra[0] == "-find":
            return ""
        assert extra == ["-property", "installationPath"]
        return str(install)

    monkeypatch.setattr(build_native, "_vswhere_output", fake_output)
    assert build_native._find_vcvarsall() == bat


def test_find_vcvarsall_scans_vs2026_layout_without_vswhere(monkeypatch, tmp_path):
    bat = tmp_path / "Microsoft Visual Studio" / "18" / "Enterprise" / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    bat.parent.mkdir(parents=True)
    bat.write_text("", encoding="utf-8")
    monkeypatch.setattr(build_native, "_vswhere_exe", lambda: None)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.delenv("ProgramW6432", raising=False)
    assert build_native._find_vcvarsall() == bat


def test_find_vcvarsall_scans_programw6432_for_a_32_bit_view(monkeypatch, tmp_path):
    x86 = tmp_path / "x86"
    native = tmp_path / "native"
    bat = native / "Microsoft Visual Studio" / "18" / "BuildTools" / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    bat.parent.mkdir(parents=True)
    bat.write_text("", encoding="utf-8")
    monkeypatch.setattr(build_native, "_vswhere_exe", lambda: None)
    monkeypatch.setenv("ProgramFiles", str(x86))
    monkeypatch.setenv("ProgramFiles(x86)", str(x86))
    monkeypatch.setenv("ProgramW6432", str(native))
    assert build_native._find_vcvarsall() == bat


def test_vswhere_output_requests_utf8(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs["encoding"]))

        class Result:
            returncode = 0
            stdout = "C:\\VS"

        return Result()

    monkeypatch.setattr(build_native.subprocess, "run", fake_run)
    found = build_native._vswhere_output(tmp_path / "vswhere.exe", ["-find", r"VC\Auxiliary\Build\vcvarsall.bat"])
    assert found == "C:\\VS"
    assert calls[0][1] == "utf-8"
    assert "-utf8" in calls[0][0]
    assert len(calls) == 1


def test_vswhere_output_retries_when_utf8_flag_is_rejected(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs["encoding"]))

        class Result:
            returncode = 1 if "-utf8" in cmd else 0
            stdout = "" if "-utf8" in cmd else "C:\\VS"

        return Result()

    monkeypatch.setattr(build_native.subprocess, "run", fake_run)
    found = build_native._vswhere_output(tmp_path / "vswhere.exe", ["-property", "installationPath"])
    assert found == "C:\\VS"
    assert "-utf8" in calls[0][0]
    assert "-utf8" not in calls[1][0]
    assert calls[1][1] == "mbcs"


def test_resolve_cl_uses_the_vcvars_path(tmp_path):
    cl_dir = tmp_path / "Hostx64" / "x64"
    cl_dir.mkdir(parents=True)
    cl = cl_dir / "cl.exe"
    cl.write_bytes(b"")
    path = os.pathsep.join((str(cl_dir), r"C:\Windows\System32"))
    assert build_native._resolve_cl({"Path": path}) == str(cl)


def test_resolve_cl_uses_the_parent_path_when_cl_is_already_visible(monkeypatch):
    def which(name):
        return r"C:\MSVC\bin\cl.exe" if name == "cl" else None

    monkeypatch.setattr(build_native.shutil, "which", which)
    assert build_native._resolve_cl(None) == r"C:\MSVC\bin\cl.exe"


def test_windows_compile_env_keeps_developer_prompt(monkeypatch):
    monkeypatch.setattr(build_native.shutil, "which", lambda name: r"C:\MSVC\bin\cl.exe" if name == "cl" else None)

    def unexpected():
        raise AssertionError("vcvarsall should not be queried when cl is already on PATH")

    monkeypatch.setattr(build_native, "_find_vcvarsall", unexpected)
    assert build_native._windows_compile_env() is None


def test_windows_compile_env_reports_a_missing_toolchain(monkeypatch):
    monkeypatch.setattr(build_native.shutil, "which", lambda name: None)
    monkeypatch.setattr(build_native, "_find_vcvarsall", lambda: None)
    with pytest.raises(SystemExit, match="cl.exe is not on PATH"):
        build_native._windows_compile_env()


def _completed(returncode=0, stdout=b"", stderr=b""):
    class Result:
        pass

    result = Result()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


def test_msvc_arch_follows_the_interpreter(monkeypatch):
    monkeypatch.setattr(build_native.platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(build_native.sys, "maxsize", 2**63)
    assert build_native._msvc_arch() == "arm64"

    monkeypatch.setattr(build_native.platform, "machine", lambda: "AMD64")
    assert build_native._msvc_arch() == "x64"

    monkeypatch.setattr(build_native.platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(build_native.sys, "maxsize", 2**31 - 1)
    assert build_native._msvc_arch() == "x86"


def test_vcvars_command_quotes_program_files_once():
    bat = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat")
    assert build_native._vcvars_command(bat, "x86") == (
        r'cmd /u /s /c "call "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools'
        r'\VC\Auxiliary\Build\vcvarsall.bat" x86 >nul && set"'
    )


def test_load_vcvars_env_reads_utf16_dump(monkeypatch, tmp_path):
    cl_dir = tmp_path / "bin"
    cl_dir.mkdir()
    (cl_dir / "cl.exe").write_bytes(b"")
    bat = tmp_path / "vcvarsall.bat"
    dumped = f"Path={cl_dir}\r\nSystemRoot=C:\\Windows\r\n"
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        assert kwargs["capture_output"] is True
        return _completed(stdout=dumped.encode("utf-16le"))

    monkeypatch.setattr(build_native.subprocess, "run", fake_run)
    monkeypatch.setattr(build_native, "_msvc_arch", lambda: "x64")
    env = build_native._load_vcvars_env(bat)
    assert captured["command"] == build_native._vcvars_command(bat, "x64")
    assert env["Path"] == str(cl_dir)
    assert env["SystemRoot"] == r"C:\Windows"


def test_load_vcvars_env_rejects_an_environment_without_cl(monkeypatch, tmp_path):
    bat = tmp_path / "vcvarsall.bat"

    def fake_run(command, **kwargs):
        dumped = "Path=C:\\Windows\\System32\r\n".encode("utf-16le")
        return _completed(stdout=dumped, stderr=b"missing toolset\n")

    monkeypatch.setattr(build_native.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="did not provide cl.exe") as exc:
        build_native._load_vcvars_env(bat)
    assert "missing toolset" in str(exc.value)


def test_load_vcvars_env_surfaces_vcvars_failure(monkeypatch, tmp_path):
    bat = tmp_path / "vcvarsall.bat"

    def fake_run(command, **kwargs):
        return _completed(returncode=1, stderr=b"Could not find the VC tools\n")

    monkeypatch.setattr(build_native.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="Failed to initialize the MSVC environment") as exc:
        build_native._load_vcvars_env(bat)
    assert "Could not find the VC tools" in str(exc.value)
