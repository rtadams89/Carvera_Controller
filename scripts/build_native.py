#!/usr/bin/env python3
"""Compile the stock-carving C extension into native/lib (in-place)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def _sources(native_dir: Path) -> list[Path]:
    names = ["profile.c", "heightmap.c", "cylindrical.c", "voxel.c", "laser.c", "mesh.c", "_stock_carve.c"]
    return [native_dir / name for name in names]


def _output_path(native_dir: Path) -> Path:
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    lib_dir = native_dir / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    return lib_dir / f"_stock_carve{suffix}"


def _cl_quote(path: Path | str) -> str:
    return '"' + str(path).replace('"', "") + '"'


def _windows_python_lib() -> Path:
    """Import library that resolves Py_* symbols when linking an extension."""
    ver = f"{sys.version_info.major}{sys.version_info.minor}"
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ""
    debug = bool(sysconfig.get_config_var("Py_DEBUG")) or ext.endswith("_d.pyd")
    name = f"python{ver}{'_d' if debug else ''}.lib"
    candidates: list[Path] = []
    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        candidates.append(Path(libdir) / name)
    candidates.append(Path(sys.base_prefix) / "libs" / name)
    for path in candidates:
        if path.is_file():
            return path
    looked = ", ".join(str(path) for path in candidates)
    raise SystemExit(f"Python import library {name} not found. Looked in: {looked}")


def _windows_cmd(native_dir: Path, include: str, out: Path, sources: list[str]) -> list[str]:
    py_lib = _windows_python_lib()
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ""
    debug = bool(sysconfig.get_config_var("Py_DEBUG")) or ext.endswith("_d.pyd")
    # /Fo must name a directory ending in a backslash. Double it so the
    # backslash does not escape the closing quote in cl's response file.
    fo = str(out.parent).replace("/", "\\").rstrip("\\") + "\\\\"
    lines = [
        "/nologo",
        "/O2",
        "/std:c11",
        "/MDd" if debug else "/MD",
        "/LD",
        "/I" + _cl_quote(include),
        "/I" + _cl_quote(native_dir),
        f'/Fo"{fo}"',
        *(_cl_quote(src) for src in sources),
        "/Fe:" + _cl_quote(out),
        "/link /LIBPATH:" + _cl_quote(py_lib.parent) + " " + py_lib.name,
    ]
    rsp = out.parent / "_stock_carve.rsp"
    rsp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Relative to cwd=native_dir so the @ argument itself has no spaces.
    return ["cl", "@lib/_stock_carve.rsp"]


def _parse_env_block(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        key = key.lstrip("\ufeff").strip()
        if sep and key:
            env[key] = value
    return env


def _env_value(env: dict[str, str], name: str) -> str:
    for key, value in env.items():
        if key.lower() == name.lower():
            return value
    return ""


def _env_exe(env: dict[str, str], exe: str) -> Path | None:
    for entry in _env_value(env, "PATH").split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry) / exe
        if candidate.is_file():
            return candidate
    return None


def _env_has_exe(env: dict[str, str], exe: str) -> bool:
    return _env_exe(env, exe) is not None


def _resolve_cl(env: dict[str, str] | None) -> str:
    """Absolute path of cl.exe.

    CreateProcess searches the calling process PATH, not the environment
    block passed to the child. vcvarsall puts cl.exe only on that block.
    """
    if env is None:
        found = shutil.which("cl")
        if found:
            return found
        raise SystemExit("cl.exe is not on PATH.")
    cl = _env_exe(env, "cl.exe")
    if cl is None:
        raise SystemExit("MSVC environment did not provide cl.exe. Install the Desktop development with C++ workload.")
    return str(cl)


def _env_roots(*names: str) -> list[str]:
    roots: list[str] = []
    for name in names:
        root = os.environ.get(name)
        if root and root not in roots:
            roots.append(root)
    return roots


def _vswhere_exe() -> Path | None:
    # The installer is a 32-bit program. ProgramW6432 covers a 32-bit Python
    # whose ProgramFiles variable points at "Program Files (x86)".
    for root in _env_roots("ProgramFiles(x86)", "ProgramFiles", "ProgramW6432"):
        path = Path(root) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if path.is_file():
            return path
    return None


def _run_vswhere(cmd: list[str], encoding: str) -> str | None:
    """Stdout from a successful vswhere query.

    None means the invocation itself failed, so the caller can retry with
    different flags. An empty string means vswhere ran and found nothing.
    """
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding=encoding,
            errors="replace",
        )
    except (OSError, LookupError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _vswhere_output(vswhere: Path, extra: list[str]) -> str:
    common = [
        str(vswhere),
        "-latest",
        "-products",
        "*",
        "-prerelease",
        "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
    ]
    # -utf8 keeps non-ASCII install paths intact. vswhere before 2.5 rejects it.
    found = _run_vswhere([*common, "-utf8", *extra], "utf-8")
    if found is not None:
        return found
    legacy = _run_vswhere([*common, *extra], "mbcs")
    return legacy or ""


def _known_vcvarsall_paths() -> list[Path]:
    # VS 2026 uses a numeric "18" directory. Older releases use the year.
    versions = ("18", "2022", "2019", "2017")
    editions = ("Enterprise", "Professional", "Community", "BuildTools", "Preview")
    paths: list[Path] = []
    # ProgramW6432 is the native Program Files directory for a 32-bit process.
    for root in _env_roots("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
        base = Path(root) / "Microsoft Visual Studio"
        for version in versions:
            for edition in editions:
                paths.append(base / version / edition / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat")
    return paths


def _find_vcvarsall() -> Path | None:
    """Locate vcvarsall.bat without assuming a Visual Studio version or edition."""
    vswhere = _vswhere_exe()
    if vswhere is not None:
        # -find works on VS 2026 layouts where installationPath can come back empty.
        found = _vswhere_output(vswhere, ["-find", r"VC\Auxiliary\Build\vcvarsall.bat"])
        for line in found.splitlines():
            path = Path(line.strip().strip('"'))
            if path.is_file():
                return path
        install = _vswhere_output(vswhere, ["-property", "installationPath"])
        for line in install.splitlines():
            path = Path(line.strip().strip('"')) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
            if path.is_file():
                return path
    for path in _known_vcvarsall_paths():
        if path.is_file():
            return path
    return None


def _msvc_arch() -> str:
    # Match the interpreter, not the host OS. A 32-bit process on ARM64 Windows
    # still needs the x86 toolset; native ARM64 Python reports arm64/aarch64.
    if sys.maxsize <= 2**32:
        return "x86"
    if platform.machine().lower() in ("arm64", "aarch64"):
        return "arm64"
    return "x64"


def _console_text(raw: bytes) -> str:
    """Decode a cmd stderr pipe.

    `cmd /u` makes the stdout `set` dump UTF-16LE. stderr stays in the console
    code page, except when the pipe itself arrives as UTF-16.
    """
    if not raw:
        return ""
    sample = raw[:8]
    if sample.startswith(b"\xff\xfe") or b"\x00" in sample:
        return raw.decode("utf-16le", errors="replace").lstrip("\ufeff").strip()
    for encoding in ("utf-8", "mbcs", "oem"):
        try:
            return raw.decode(encoding).strip()
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace").strip()


def _failure_detail(raw: bytes, limit: int = 800) -> str:
    text = _console_text(raw)
    if not text:
        return ""
    if len(text) > limit:
        text = text[-limit:]
    return "\n" + text


def _vcvars_command(vcvarsall: Path, arch: str) -> str:
    # /s strips one outer pair of quotes, so the path must be quoted exactly
    # once. Doubling them leaves cmd executing ""C:\Program and stopping at
    # the space in "Program Files". /u makes the internal `set` write UTF-16LE.
    # The bat's stdout is discarded so it cannot corrupt that dump.
    return f'cmd /u /s /c "call "{vcvarsall}" {arch} >nul && set"'


def _load_vcvars_env(vcvarsall: Path) -> dict[str, str]:
    arch = _msvc_arch()
    command = _vcvars_command(vcvarsall, arch)
    try:
        completed = subprocess.run(command, check=False, capture_output=True)
    except OSError as exc:
        raise SystemExit(f"Failed to initialize the MSVC environment with {vcvarsall}.\n{exc}") from exc
    detail = _failure_detail(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(f"Failed to initialize the MSVC environment with {vcvarsall}.{detail}")
    env = _parse_env_block(completed.stdout.decode("utf-16le", errors="replace"))
    if not _env_has_exe(env, "cl.exe"):
        raise SystemExit(
            f"MSVC environment from {vcvarsall} did not provide cl.exe. "
            "Install the Desktop development with C++ workload."
            f"{detail}"
        )
    return env


def _windows_compile_env() -> dict[str, str] | None:
    """Environment in which `cl` can run.

    GitHub-hosted Windows images install MSVC but leave it off PATH until
    vcvarsall.bat is applied. A Developer Command Prompt already has `cl`.
    """
    if shutil.which("cl"):
        return None
    vcvarsall = _find_vcvarsall()
    if vcvarsall is None:
        raise SystemExit(
            "cl.exe is not on PATH and the MSVC toolchain was not found. "
            "Install Visual Studio Build Tools with the Desktop development with C++ workload."
        )
    print(f"loading MSVC environment from {vcvarsall}")
    return _load_vcvars_env(vcvarsall)


def build() -> Path:
    native_dir = Path(__file__).resolve().parents[1] / "carveracontroller" / "addons" / "stock" / "simulator" / "native"
    include = sysconfig.get_config_var("INCLUDEPY")
    if not include:
        raise SystemExit("Python headers not found (INCLUDEPY). Install python3-dev.")
    out = _output_path(native_dir)
    sources = [str(path) for path in _sources(native_dir)]
    compile_env = None
    if sys.platform == "win32":
        # Resolve the toolchain before writing the response file, so a missing
        # MSVC install does not leave _stock_carve.rsp behind.
        compile_env = _windows_compile_env()
        cl = _resolve_cl(compile_env)
        cmd = _windows_cmd(native_dir, include, out, sources)
        cmd[0] = cl
    else:
        cc = os.environ.get("CC", "cc")
        cflags = os.environ.get("CFLAGS", "")
        # Darwin extensions are Mach-O bundles. -shared yields a dylib whose
        # Py_* symbols fail to link, or that CPython refuses to load.
        if sys.platform == "darwin":
            link_flags = ["-bundle", "-undefined", "dynamic_lookup"]
        else:
            link_flags = ["-shared"]
        cmd = [
            cc,
            "-O3",
            "-std=c99",
            "-fPIC",
            *link_flags,
            "-Wall",
            "-Wextra",
            f"-I{include}",
            f"-I{native_dir}",
            *sources,
            "-o",
            str(out),
            "-lm",
        ]
        if cflags:
            cmd[1:1] = cflags.split()
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=native_dir, env=compile_env)
    if sys.platform == "win32":
        (native_dir / "lib" / "_stock_carve.rsp").unlink(missing_ok=True)
    print(f"built {out}")
    return out


if __name__ == "__main__":
    build()
