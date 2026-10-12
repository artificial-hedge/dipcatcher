#!/usr/bin/env python3
"""Offline installed-wheel smoke for the dipcatcher console scripts.

Builds a wheel from the working tree, installs it into a clean temporary
virtualenv without network, runs the launcher from a neutral directory, then
uninstalls it. The JSON record is evidence for this machine only.

Observed with uv 0.13.0 and CPython 3.12.3 before this script relied on it:

* ``uv build --wheel --offline`` succeeds. Hatchling 1.32.4 is already in
  the uv wheel cache, so build isolation does not need network.
* ``uv build --offline --no-build-isolation`` fails because hatchling is
  not installed in the project ``.venv``.
* ``uv pip install --offline <wheel>`` fails closed. The uv simple-index
  cache on this image only has the build backend, so the resolver reports
  runtime dependencies as missing even when their wheels are under
  ``wheels-v7``.
* ``uv sync --frozen --offline --no-dev --no-install-project`` installs
  that locked closure from the wheel cache. ``uv pip install --offline
  --no-deps <wheel>`` then installs the built wheel.
* If the lockfile install is unavailable, a ``.pth`` file can expose the
  project ``.venv`` site-packages for third-party imports. ``fx1``,
  ``quant_fund``, and ``dipcatcher_cli`` must still resolve from the wheel.

The per-run ``install_notes`` record which of these steps actually ran.
Signing, rollback drills, and the macOS/Windows platform matrix are not
covered.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

SCHEMA = "dipcatcher.release_smoke.v1"
SCOPE = (
    "Offline installed-wheel smoke on this machine. "
    "Signing, rollback drills, and the platform matrix (macOS/Windows) are not covered."
)
NOT_COVERED = (
    "signing",
    "rollback drills",
    "platform matrix (macOS/Windows)",
)
STRATEGY = (
    "uv build --wheel --offline; if build dependencies are unavailable offline, "
    "uv build --wheel --offline --no-build-isolation",
    "uv pip install --offline <wheel> into a clean virtualenv",
    "if that resolver cannot see the cached closure: "
    "uv sync --frozen --offline --no-dev --no-install-project, then "
    "uv pip install --offline --no-deps <wheel>",
    "if the lockfile install fails: a .pth file adds the project virtualenv "
    "site-packages for third-party imports only",
)
RETRY_DELAY_S = 30.0
OWNED = ("fx1", "quant_fund", "dipcatcher_cli")
CONSOLE_SCRIPTS = ("dipcatcher", "quant", "fxi", "fx1", "verify-ledger", "mc-engine")
MODE_UV_PIP = "uv-pip-install-offline"
MODE_LOCKFILE = "uv-sync-frozen-offline-deps+uv-pip-offline-no-deps"
MODE_PTH = "pth-project-venv-site-packages"
MODE_NONE = "not-installed"
BUILD_ISOLATED = "uv-build-offline-isolated"
BUILD_NO_ISOLATION = "uv-build-offline-no-build-isolation"
_BUILD_TIMEOUT_S = 180.0
_INSTALL_TIMEOUT_S = 180.0
_HELP_TIMEOUT_S = 180.0
_EXEC_TIMEOUT_S = 90.0
_PROBE = (
    "import json, sys, sysconfig\n"
    "marker = 'dipcatcher.release_smoke.probe'\n"
    "names = ('fx1', 'quant_fund', 'dipcatcher_cli')\n"
    "mods = {}\n"
    "for name in names:\n"
    "    try:\n"
    "        mod = __import__(name)\n"
    "    except Exception as exc:\n"
    "        mods[name] = {'error': type(exc).__name__}\n"
    "        continue\n"
    "    mods[name] = {'file': getattr(mod, '__file__', '') or ''}\n"
    "print(json.dumps({'marker': marker, 'modules': mods, "
    "'purelib': sysconfig.get_path('purelib'), 'prefix': sys.prefix}))\n"
)
_PURELIB = (
    "import sysconfig; print(sysconfig.get_path('purelib'))  # dipcatcher.release_smoke.purelib"
)
_IMPORT_PROBE = "import dipcatcher_cli"
_IMPORT_MARKERS = ("ModuleNotFoundError", "ImportError", "SyntaxError", "IndentationError")
_IMPORT_ERRORS = frozenset(_IMPORT_MARKERS)
_OFFLINE_MARKERS = (
    "network was disabled",
    "network connectivity is disabled",
    "failed to download",
    "because the network",
    "client error (connect)",
    "temporary failure in name resolution",
    "nodename nor servname",
    "executable not found:",
)
_EXEC_NAMES = frozenset({"dipcatcher_exec_fx1", "dipcatcher_exec_fx1_lite"})
_CHECK_FIELDS = ("name", "command", "exit", "passed", "detail")


class CommandRunner(Protocol):
    """Run one command and capture its text output."""

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_s: float,
        stdin: bytes = b"",
    ) -> CommandResult: ...


class _Loopback(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    @property
    def url(self) -> str: ...


@dataclass(frozen=True)
class CommandResult:
    """One finished command. ``returncode`` is -1 when the timeout fired."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class CheckSpec:
    """One smoke command and the conditions that make it pass."""

    name: str
    args: tuple[str, ...]
    expect_exit: int = 0
    stdout_contains: tuple[str, ...] = ()
    stdout_exact: str | None = None
    stdin: bytes = b""
    timeout_s: float = _HELP_TIMEOUT_S
    extra_env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Layout:
    """Scratch directories for one smoke run."""

    work: Path
    wheel: Path
    neutral: Path
    home: Path
    config: Path
    sessions: Path


@dataclass
class InstallHit:
    """Result of one install strategy."""

    ok: bool
    mode: str
    notes: list[str]
    venv: Path | None
    modules: dict[str, str]
    modules_from_install: bool
    probe_detail: str = ""
    probe_import_failure: bool = False


@dataclass
class Attempt:
    """One build, install, check, and uninstall pass."""

    wheel_name: str = ""
    wheel_sha256: str = ""
    wheel_path: Path | None = None
    build_status: str = "failed"
    build_mode: str = ""
    install_mode: str = MODE_NONE
    notes: list[str] = field(default_factory=list)
    modules: dict[str, str] = field(default_factory=dict)
    modules_from_install: bool = False
    checks: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class Options:
    """CLI options, with defaults resolved."""

    out: Path | None
    checksums: Path | None
    repo: Path
    work_dir: Path | None
    uv: Path
    python: Path


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_sha256sums(entries: Mapping[str, str]) -> str:
    """GNU ``sha256sum`` text, names sorted, two spaces, trailing newline."""
    lines = [f"{entries[name]}  {name}" for name in sorted(entries)]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    """Atomically replace ``path`` and set ``mode`` (default 0600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    wrote = False
    try:
        _write_all(fd, data)
        os.fsync(fd)
        wrote = True
    finally:
        os.close(fd)
        if not wrote:
            tmp.unlink(missing_ok=True)
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    os.chmod(path, mode)


def atomic_write_text(path: Path, text: str, mode: int = 0o600) -> None:
    """Write text atomically as UTF-8 with ``mode``."""
    atomic_write(path, text.encode("utf-8"), mode)


def atomic_write_json(path: Path, payload: Mapping[str, object], mode: int = 0o600) -> None:
    """Write one JSON document atomically with ``mode``."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, text, mode)


def write_checksum_file(path: Path, artifacts: Sequence[tuple[str, Path]]) -> dict[str, str]:
    """Write SHA256SUMS for ``artifacts`` as ``(filename, path)`` pairs."""
    entries: dict[str, str] = {}
    for name, artifact in artifacts:
        if name in entries:
            raise ValueError(f"duplicate checksum name {name}")
        entries[name] = sha256_file(artifact)
    atomic_write_text(path, format_sha256sums(entries))
    return entries


def scrub(text: str, secrets: Sequence[str]) -> str:
    """Replace each secret of length >= 4 with ``[redacted]``."""
    redacted = text
    for secret in secrets:
        if len(secret) >= 4:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted


def bound(text: str, limit: int = 400) -> str:
    """Keep at most ``limit`` characters."""
    if len(text) <= limit:
        return text
    return text[:limit]


def tail_text(text: str, limit: int = 400) -> str:
    """Collapse whitespace and keep the last ``limit`` characters."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[-limit:]


def utc_stamp(moment: datetime) -> str:
    """Format an aware datetime as UTC ``YYYY-MM-DDTHH:MM:SSZ``."""
    if moment.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def platform_facts() -> dict[str, str]:
    """Platform fields recorded in the evidence document."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def wheel_version(filename: str) -> str:
    """Return the version component of a wheel filename, or ``\"\"``."""
    name = Path(filename).name
    if not name.endswith(".whl"):
        return ""
    parts = name[: -len(".whl")].split("-")
    if len(parts) < 5:
        return ""
    return parts[1]


def _text(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    return data.decode("utf-8", "replace")


def subprocess_runner(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float,
    stdin: bytes = b"",
) -> CommandResult:
    """Run ``args`` with a pipe on stdin so the child is not a TTY."""
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            env=dict(env),
            input=stdin,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _text(exc.stdout)
        stderr = _text(exc.stderr) + "\n[timed out]"
        return CommandResult(-1, stdout, stderr, True)
    return CommandResult(completed.returncode, _text(completed.stdout), _text(completed.stderr))


def run_argv(
    runner: CommandRunner,
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float,
    stdin: bytes = b"",
) -> CommandResult:
    """Run ``args``, converting a missing executable into a result."""
    try:
        return runner(args, cwd=cwd, env=env, timeout_s=timeout_s, stdin=stdin)
    except FileNotFoundError as exc:
        name = exc.filename or (args[0] if args else "")
        return CommandResult(127, "", f"executable not found: {name}", False)


def is_import_text(text: str) -> bool:
    """True when ``text`` names an import or syntax failure."""
    return any(marker in text for marker in _IMPORT_MARKERS)


def _judge(spec: CheckSpec, result: CommandResult) -> tuple[bool, str]:
    if result.timed_out:
        return False, "timed out"
    if result.returncode != spec.expect_exit:
        return False, f"exit {result.returncode}: {tail_text(result.stderr or result.stdout)}"
    if spec.stdout_exact is not None and result.stdout.strip() != spec.stdout_exact:
        return False, "stdout mismatch: " + bound(result.stdout.strip(), 200)
    missing = [item for item in spec.stdout_contains if item not in result.stdout]
    if missing:
        return False, "stdout missing " + ", ".join(missing)
    return True, ""


def check_row(
    name: str,
    command: Sequence[str],
    *,
    exit_code: int,
    passed: bool,
    detail: str,
    import_failure: bool = False,
) -> dict[str, object]:
    """One internal check row. ``import_failure`` is stripped before emit."""
    return {
        "name": name,
        "command": [str(part) for part in command],
        "exit": exit_code,
        "passed": passed,
        "detail": detail,
        "import_failure": import_failure and not passed,
    }


def evaluate_check(
    spec: CheckSpec, result: CommandResult, secrets: Sequence[str]
) -> dict[str, object]:
    """Judge ``result`` against ``spec`` and scrub secrets from the detail."""
    passed, detail = _judge(spec, result)
    text = result.stderr + "\n" + result.stdout
    return check_row(
        spec.name,
        spec.args,
        exit_code=result.returncode,
        passed=passed,
        detail=scrub(detail, secrets),
        import_failure=not passed and is_import_text(text),
    )


def import_retry_reason(checks: Sequence[Mapping[str, object]]) -> str:
    """Name of the first failed check that looks like a broken import, or ``\"\"``."""
    for row in checks:
        if row.get("passed"):
            continue
        if row.get("import_failure"):
            return str(row.get("name", "check"))
    return ""


def public_check(row: Mapping[str, object], secrets: Sequence[str] = ()) -> dict[str, object]:
    """The check fields stored in the evidence document."""
    published = {key: row[key] for key in _CHECK_FIELDS}
    detail = published["detail"]
    text = detail if isinstance(detail, str) else str(detail)
    published["detail"] = scrub(text, secrets)
    return published


def _as_dict(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return value
    return None


def _module_entry(entry: object) -> tuple[str, str]:
    payload = _as_dict(entry)
    if payload is None:
        return "", ""
    file_name = payload.get("file")
    error = payload.get("error")
    path = file_name if isinstance(file_name, str) else ""
    err = error if isinstance(error, str) else ""
    return path, err


def _module_files(modules: object) -> tuple[dict[str, str], str]:
    payload = _as_dict(modules)
    if payload is None:
        return {}, ""
    files: dict[str, str] = {}
    problem = ""
    for name in OWNED:
        path, error = _module_entry(payload.get(name))
        files[name] = path
        if error and not problem:
            problem = error
    return files, problem


def parse_probe(stdout: str) -> tuple[dict[str, str], str, str]:
    """Parse the module probe into ``(files, purelib, error type)``."""
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        return {}, "", ""
    payload = _as_dict(value)
    if payload is None:
        return {}, "", ""
    purelib = payload.get("purelib")
    root = purelib if isinstance(purelib, str) else ""
    files, problem = _module_files(payload.get("modules"))
    return files, root, problem


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _outside_root(files: Mapping[str, str], root: Path) -> str:
    for path in files.values():
        if not path or not _is_under(Path(path), root):
            return path or "<empty>"
    return ""


def modules_match(files: Mapping[str, str], root: Path) -> bool:
    """True when each owned package file lives under ``root``."""
    if set(files) != set(OWNED):
        return False
    return _outside_root(files, root) == ""


def note_from(label: str, result: CommandResult) -> str:
    """One scrubbable line describing a command result."""
    status = "ok" if result.returncode == 0 else f"exit {result.returncode}"
    detail = tail_text(result.stderr or result.stdout)
    if not detail:
        return f"{label}: {status}"
    return f"{label}: {status}: {detail}"


def build_failure_status(text: str) -> str:
    """``offline-unavailable`` when the failure is environmental, else ``failed``."""
    lowered = text.lower()
    if any(marker in lowered for marker in _OFFLINE_MARKERS):
        return "offline-unavailable"
    return "failed"


def default_uv() -> Path:
    """Resolve the uv binary without requiring it to be on PATH."""
    found = shutil.which("uv")
    if found:
        return Path(found)
    for candidate in (Path.home() / ".local" / "bin" / "uv", Path("/home/ubuntu/.local/bin/uv")):
        if candidate.is_file():
            return candidate
    return Path("uv")


def project_python(repo: Path) -> Path:
    """The project virtualenv interpreter, or the current one."""
    for parts in (("bin", "python"), ("Scripts", "python.exe")):
        candidate = repo.joinpath(".venv", *parts)
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def scripts_dir(venv: Path) -> Path:
    """Console-script directory for ``venv``."""
    if os.name == "nt":
        return venv / "Scripts"
    return venv / "bin"


def venv_python(venv: Path) -> Path:
    """Interpreter inside ``venv``, even if the file is not created yet."""
    for parts in (("bin", "python"), ("Scripts", "python.exe")):
        candidate = venv.joinpath(*parts)
        if candidate.is_file():
            return candidate
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def console_script(venv: Path, name: str) -> Path:
    """Path of an installed console script, preferring one that exists."""
    folder = scripts_dir(venv)
    for suffix in ("", ".exe", ".cmd"):
        candidate = folder / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    return folder / name


def project_sites(repo: Path) -> list[Path]:
    """Site-packages directories of the project virtualenv, if it exists."""
    root = repo / ".venv"
    found: list[Path] = []
    lib = root / "lib"
    if lib.is_dir():
        found.extend(path for path in lib.glob("python*/site-packages") if path.is_dir())
    windows = root / "Lib" / "site-packages"
    if windows.is_dir():
        found.append(windows)
    return found


def owned_package_dirs(repo: Path) -> set[str]:
    """Installed ``fx1`` / ``dipcatcher_cli`` trees in the project environment.

    An editable ``.pth`` does not create these directories. A wheel install into
    the shared project environment does, and this smoke must not do that.
    """
    found: set[str] = set()
    for site in project_sites(repo):
        for name in ("fx1", "dipcatcher_cli"):
            init = site / name / "__init__.py"
            if init.is_file():
                found.add(str(init.resolve()))
    return found


def uv_command_env(venv: Path | None = None) -> dict[str, str]:
    """Environment for uv. The project cache stays available; the parent env does not leak."""
    env: dict[str, str] = {}
    for key in ("PATH", "HOME", "USER", "LANG", "LC_ALL", "UV_CACHE_DIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    env.setdefault("PATH", os.pathsep.join(["/usr/local/bin", "/usr/bin", "/bin"]))
    env.setdefault("HOME", str(Path.home()))
    env["UV_OFFLINE"] = "1"
    if venv is not None:
        env["UV_PROJECT_ENVIRONMENT"] = str(venv)
    return env


def child_env(layout: Layout, venv: Path, extra: Mapping[str, str]) -> dict[str, str]:
    """Environment for an installed console script. Not a TTY, no project source tree."""
    scripts = str(scripts_dir(venv))
    env = {
        "PATH": os.pathsep.join([scripts, "/usr/local/bin", "/usr/bin", "/bin"]),
        "HOME": str(layout.home),
        "VIRTUAL_ENV": str(venv),
        "FX1_CONFIG_DIR": str(layout.config),
        "FX1_SESSION_DIR": str(layout.sessions),
        "XDG_CACHE_HOME": str(layout.home / ".cache"),
        "XDG_CONFIG_HOME": str(layout.home / ".config"),
        "MPLCONFIGDIR": str(layout.home / ".mpl"),
        "NO_COLOR": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    lc_all = os.environ.get("LC_ALL")
    if lc_all:
        env["LC_ALL"] = lc_all
    env.update(extra)
    return env


def make_layout(work: Path) -> Layout:
    """Create the scratch directories used by one smoke run."""
    layout = Layout(
        work=work,
        wheel=work / "wheel",
        neutral=work / "neutral",
        home=work / "home",
        config=work / "config",
        sessions=work / "sessions",
    )
    for path in (layout.wheel, layout.neutral, layout.home, layout.config, layout.sessions):
        path.mkdir(parents=True, exist_ok=True)
    for name in (".cache", ".config", ".mpl"):
        (layout.home / name).mkdir(exist_ok=True)
    return layout


def _build_command(
    uv: Path, repo: Path, out_dir: Path, python: Path, *, no_isolation: bool
) -> list[str]:
    command = [
        str(uv),
        "build",
        "--wheel",
        "--offline",
        "--no-python-downloads",
        "--out-dir",
        str(out_dir),
        "--no-create-gitignore",
        "--python",
        str(python),
    ]
    if no_isolation:
        command.append("--no-build-isolation")
    command.append(str(repo))
    return command


def newest_wheel(directory: Path) -> Path | None:
    """The newest ``*.whl`` in ``directory``, if any."""
    wheels = sorted(directory.glob("*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        return None
    return wheels[-1]


def _wheel_if_ok(result: CommandResult, out_dir: Path) -> Path | None:
    if result.returncode != 0:
        return None
    return newest_wheel(out_dir)


def prepare_out(directory: Path) -> None:
    """Clear previous wheels so a retry cannot hash a stale file."""
    if directory.exists():
        for child in directory.glob("*.whl"):
            child.unlink()
    directory.mkdir(parents=True, exist_ok=True)


def build_wheel(
    uv: Path,
    repo: Path,
    out_dir: Path,
    python: Path,
    runner: CommandRunner,
) -> tuple[Path | None, str, str, list[str]]:
    """Build one wheel offline. Returns ``(path, status, mode, notes)``."""
    notes: list[str] = []
    prepare_out(out_dir)
    env = uv_command_env()
    first = run_argv(
        runner,
        _build_command(uv, repo, out_dir, python, no_isolation=False),
        cwd=repo,
        env=env,
        timeout_s=_BUILD_TIMEOUT_S,
    )
    notes.append(note_from("uv build --wheel --offline", first))
    wheel = _wheel_if_ok(first, out_dir)
    if wheel is not None:
        return wheel, "ok", BUILD_ISOLATED, notes
    second = run_argv(
        runner,
        _build_command(uv, repo, out_dir, python, no_isolation=True),
        cwd=repo,
        env=env,
        timeout_s=_BUILD_TIMEOUT_S,
    )
    notes.append(note_from("uv build --offline --no-build-isolation", second))
    wheel = _wheel_if_ok(second, out_dir)
    if wheel is not None:
        return wheel, "ok", BUILD_NO_ISOLATION, notes
    text = "\n".join((first.stderr, first.stdout, second.stderr, second.stdout))
    return None, build_failure_status(text), "", notes


def drop_tree(path: Path | None) -> None:
    """Remove a temporary tree if it exists."""
    if path is not None and path.exists():
        shutil.rmtree(path, ignore_errors=True)


def create_venv(
    uv: Path, dest: Path, python: Path, runner: CommandRunner, cwd: Path
) -> CommandResult:
    """Create a clean virtualenv at ``dest`` with the existing interpreter."""
    drop_tree(dest)
    return run_argv(
        runner,
        [
            str(uv),
            "venv",
            str(dest),
            "--python",
            str(python),
            "--offline",
            "--no-project",
            "--no-python-downloads",
        ],
        cwd=cwd,
        env=uv_command_env(),
        timeout_s=_INSTALL_TIMEOUT_S,
    )


def _pip_install(
    uv: Path, python: Path, wheel: Path, runner: CommandRunner, cwd: Path, *, no_deps: bool
) -> CommandResult:
    command = [str(uv), "pip", "install", "--offline", "--python", str(python)]
    if no_deps:
        command.append("--no-deps")
    command.append(str(wheel))
    return run_argv(runner, command, cwd=cwd, env=uv_command_env(), timeout_s=_INSTALL_TIMEOUT_S)


def _probe_modules(python: Path, runner: CommandRunner, cwd: Path) -> CommandResult:
    return run_argv(
        runner,
        [str(python), "-c", _PROBE],
        cwd=cwd,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONDONTWRITEBYTECODE": "1"},
        timeout_s=60.0,
    )


def _purelib(python: Path, runner: CommandRunner, cwd: Path) -> str:
    result = run_argv(
        runner,
        [str(python), "-c", _PURELIB],
        cwd=cwd,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        timeout_s=30.0,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def probe_hit(result: CommandResult) -> tuple[dict[str, str], bool, bool, str]:
    """Interpret a module probe: files, matched, import-failure, detail."""
    files, root, problem = parse_probe(result.stdout)
    if result.returncode != 0:
        text = result.stderr + "\n" + result.stdout
        return files, False, is_import_text(text), tail_text(text) or "module probe failed"
    matched = bool(root) and modules_match(files, Path(root))
    detail = "" if matched else "owned modules are not inside the install"
    return files, matched, problem in _IMPORT_ERRORS, detail


def _failed_hit(mode: str, notes: list[str], detail: str, *, import_failure: bool) -> InstallHit:
    return InstallHit(False, mode, notes, None, {}, False, detail, import_failure)


def _try_pip_offline(
    uv: Path, python: Path, wheel: Path, work: Path, runner: CommandRunner, cwd: Path
) -> InstallHit:
    notes: list[str] = []
    venv = work / "venv-pip"
    created = create_venv(uv, venv, python, runner, cwd)
    notes.append(note_from("uv venv", created))
    if created.returncode != 0:
        return _failed_hit(
            MODE_UV_PIP, notes, "could not create a virtualenv", import_failure=False
        )
    installed = _pip_install(uv, venv_python(venv), wheel, runner, cwd, no_deps=False)
    notes.append(note_from("uv pip install --offline", installed))
    if installed.returncode != 0:
        return _failed_hit(
            MODE_UV_PIP, notes, "offline dependency resolve failed", import_failure=False
        )
    return _hit_from_probe(MODE_UV_PIP, notes, venv, runner, cwd)


def _sync_deps(
    uv: Path, repo: Path, venv: Path, python: Path, runner: CommandRunner
) -> CommandResult:
    return run_argv(
        runner,
        [
            str(uv),
            "sync",
            "--frozen",
            "--offline",
            "--no-dev",
            "--no-install-project",
            "--no-python-downloads",
            "--python",
            str(python),
            "--directory",
            str(repo),
        ],
        cwd=repo,
        env=uv_command_env(venv),
        timeout_s=_INSTALL_TIMEOUT_S,
    )


def _try_lockfile(
    uv: Path, repo: Path, python: Path, wheel: Path, work: Path, runner: CommandRunner, cwd: Path
) -> InstallHit:
    notes: list[str] = []
    venv = work / "venv-sync"
    drop_tree(venv)
    synced = _sync_deps(uv, repo, venv, python, runner)
    notes.append(note_from("uv sync --frozen --offline --no-dev --no-install-project", synced))
    if synced.returncode != 0:
        return _failed_hit(MODE_LOCKFILE, notes, "lockfile install failed", import_failure=False)
    installed = _pip_install(uv, venv_python(venv), wheel, runner, cwd, no_deps=True)
    notes.append(note_from("uv pip install --offline --no-deps", installed))
    if installed.returncode != 0:
        return _failed_hit(MODE_LOCKFILE, notes, "wheel install failed", import_failure=False)
    return _hit_from_probe(MODE_LOCKFILE, notes, venv, runner, cwd)


def write_third_party_pth(site: Path, project_site: Path) -> Path:
    """Point ``site`` at ``project_site`` for third-party imports only."""
    site.mkdir(parents=True, exist_ok=True)
    target = site / "zz-third-party.pth"
    text = (
        "# third-party dependencies from the project environment\n"
        "# fx1, quant_fund, and dipcatcher_cli must resolve from this install\n"
        f"{project_site}\n"
    )
    target.write_text(text, encoding="utf-8")
    return target


def _try_pth(
    uv: Path,
    project_python_path: Path,
    wheel: Path,
    work: Path,
    runner: CommandRunner,
    cwd: Path,
) -> InstallHit:
    notes: list[str] = []
    venv = work / "venv-pth"
    created = create_venv(uv, venv, project_python_path, runner, cwd)
    notes.append(note_from("uv venv (pth fallback)", created))
    if created.returncode != 0:
        return _failed_hit(MODE_PTH, notes, "could not create a virtualenv", import_failure=False)
    py = venv_python(venv)
    installed = _pip_install(uv, py, wheel, runner, cwd, no_deps=True)
    notes.append(note_from("uv pip install --offline --no-deps", installed))
    if installed.returncode != 0:
        return _failed_hit(MODE_PTH, notes, "wheel install failed", import_failure=False)
    project_site = _purelib(project_python_path, runner, cwd)
    venv_site = _purelib(py, runner, cwd)
    if not project_site or not venv_site:
        return _failed_hit(MODE_PTH, notes, "could not locate site-packages", import_failure=False)
    write_third_party_pth(Path(venv_site), Path(project_site))
    notes.append("wrote zz-third-party.pth for third-party imports")
    return _hit_from_probe(MODE_PTH, notes, venv, runner, cwd)


def _hit_from_probe(
    mode: str, notes: list[str], venv: Path, runner: CommandRunner, cwd: Path
) -> InstallHit:
    probed = _probe_modules(venv_python(venv), runner, cwd)
    files, matched, import_failure, detail = probe_hit(probed)
    notes.append(note_from("module probe", probed))
    if not matched:
        return InstallHit(False, mode, notes, None, files, False, detail, import_failure)
    return InstallHit(True, mode, notes, venv, files, True, "", False)


def _carry(hit: InstallHit, prior: InstallHit) -> InstallHit:
    hit.notes = prior.notes + hit.notes
    hit.probe_import_failure = hit.probe_import_failure or prior.probe_import_failure
    return hit


def install_wheel(
    uv: Path,
    repo: Path,
    python: Path,
    wheel: Path,
    work: Path,
    runner: CommandRunner,
    cwd: Path,
) -> InstallHit:
    """Try offline pip, then the lockfile cache, then a ``.pth`` fallback."""
    pip = _try_pip_offline(uv, python, wheel, work, runner, cwd)
    if pip.ok:
        return pip
    drop_tree(work / "venv-pip")
    locked = _carry(_try_lockfile(uv, repo, python, wheel, work, runner, cwd), pip)
    if locked.ok:
        return locked
    drop_tree(work / "venv-sync")
    return _carry(_try_pth(uv, python, wheel, work, runner, cwd), locked)


def _version_spec(script: str, version: str) -> CheckSpec:
    if version:
        return CheckSpec(
            "dipcatcher_version",
            (script, "version"),
            stdout_exact=f"dipcatcher {version}",
            timeout_s=60.0,
        )
    return CheckSpec(
        "dipcatcher_version",
        (script, "version"),
        stdout_contains=("dipcatcher ",),
        timeout_s=60.0,
    )


def make_cli_checks(
    venv: Path,
    *,
    version: str,
    url: str,
    fx1_key: str,
    lite_key: str,
) -> list[CheckSpec]:
    """Console-script checks run from a neutral directory."""
    dip = str(console_script(venv, "dipcatcher"))
    quant = str(console_script(venv, "quant"))
    fxi = str(console_script(venv, "fxi"))
    py = str(venv_python(venv))
    endpoint = (
        ("DIPCATCHER_FX1_API_KEY", fx1_key),
        ("DIPCATCHER_FX1_BASE_URL", url),
        ("DIPCATCHER_FX1_LITE_API_KEY", lite_key),
        ("DIPCATCHER_FX1_LITE_BASE_URL", url),
    )
    return [
        CheckSpec(
            "dipcatcher_help", (dip, "--help"), stdout_contains=("Start a conversation", "Usage:")
        ),
        _version_spec(dip, version),
        CheckSpec(
            "dipcatcher_exec_help", (dip, "exec", "--help"), stdout_contains=("dipcatcher exec",)
        ),
        CheckSpec(
            "dipcatcher_models_no_endpoint",
            (dip, "models"),
            stdout_contains=("fx1", "fx1-lite", "unset"),
        ),
        CheckSpec(
            "dipcatcher_exec_fx1",
            (dip, "exec", "--model", "fx1", "hello"),
            stdout_exact="stub fx1 reply",
            timeout_s=_EXEC_TIMEOUT_S,
            extra_env=endpoint,
        ),
        CheckSpec(
            "dipcatcher_exec_fx1_lite",
            (dip, "exec", "--model", "fx1-lite", "hello"),
            stdout_exact="stub fx1-lite reply",
            timeout_s=_EXEC_TIMEOUT_S,
            extra_env=endpoint,
        ),
        CheckSpec("quant_help", (quant, "--help"), stdout_contains=("Usage:",)),
        CheckSpec("fxi_help", (fxi, "--help"), stdout_contains=("Usage:", "fxi")),
        CheckSpec("dipcatcher_doctor_help", (dip, "doctor", "--help"), stdout_contains=("doctor",)),
        CheckSpec(
            "dipcatcher_bare_nontty",
            (dip,),
            stdin=b"",
            stdout_contains=("Start a conversation", "Usage:"),
        ),
        CheckSpec(
            "python_m_dipcatcher_cli_help",
            (py, "-m", "dipcatcher_cli", "--help"),
            stdout_contains=("Usage:",),
        ),
    ]


def run_one(
    spec: CheckSpec,
    runner: CommandRunner,
    base_env: Mapping[str, str],
    cwd: Path,
    secrets: Sequence[str],
) -> dict[str, object]:
    """Run one check spec."""
    env = dict(base_env)
    env.update(dict(spec.extra_env))
    result = run_argv(
        runner, spec.args, cwd=cwd, env=env, timeout_s=spec.timeout_s, stdin=spec.stdin
    )
    return evaluate_check(spec, result, secrets)


def run_planned(
    specs: Sequence[CheckSpec],
    runner: CommandRunner,
    base_env: Mapping[str, str],
    cwd: Path,
    secrets: Sequence[str],
    *,
    url_ok: bool,
    stub_error: str,
    stub_import: bool,
) -> list[dict[str, object]]:
    """Run planned checks. Exec rows fail closed when the stub did not start."""
    rows: list[dict[str, object]] = []
    for spec in specs:
        if spec.name in _EXEC_NAMES and not url_ok:
            rows.append(
                check_row(
                    spec.name,
                    spec.args,
                    exit_code=1,
                    passed=False,
                    detail=stub_error or "loopback stub did not start",
                    import_failure=stub_import,
                )
            )
            continue
        rows.append(run_one(spec, runner, base_env, cwd, secrets))
    return rows


def scripts_still_present(venv: Path) -> list[str]:
    """Console-script filenames that are still on disk."""
    found: list[str] = []
    folder = scripts_dir(venv)
    for name in CONSOLE_SCRIPTS:
        for suffix in ("", ".exe", ".cmd"):
            candidate = folder / f"{name}{suffix}"
            if candidate.is_file():
                found.append(candidate.name)
    return found


def _not_installed_rows() -> list[dict[str, object]]:
    detail = "install did not complete"
    command = ["uv", "pip", "uninstall", "fx-1"]
    return [
        check_row("console_scripts_removed", command, exit_code=1, passed=False, detail=detail),
        check_row(
            "owned_import_removed",
            ["python", "-c", _IMPORT_PROBE],
            exit_code=1,
            passed=False,
            detail=detail,
        ),
    ]


def removal_checks(
    venv: Path, uv: Path, runner: CommandRunner, cwd: Path, secrets: Sequence[str]
) -> list[dict[str, object]]:
    """Uninstall ``fx-1`` and require the console scripts and import to be gone."""
    py = venv_python(venv)
    result = run_argv(
        runner,
        [str(uv), "pip", "uninstall", "--python", str(py), "fx-1"],
        cwd=cwd,
        env=uv_command_env(),
        timeout_s=60.0,
    )
    present = scripts_still_present(venv)
    removed = result.returncode == 0 and not present
    detail = "" if removed else note_from("uv pip uninstall fx-1", result)
    if present:
        detail = (detail + " still present: " + ", ".join(present)).strip()
    rows = [
        check_row(
            "console_scripts_removed",
            [str(uv), "pip", "uninstall", "--python", str(py), "fx-1"],
            exit_code=0 if removed else result.returncode,
            passed=removed,
            detail=scrub(detail, secrets),
        )
    ]
    spec = CheckSpec(
        "owned_import_removed", (str(py), "-c", _IMPORT_PROBE), expect_exit=1, timeout_s=60.0
    )
    rows.append(
        run_one(
            spec, runner, {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}, cwd, secrets
        )
    )
    return rows


def leak_rows(repo: Path, before: set[str]) -> list[dict[str, object]]:
    """Fail if this smoke installed owned packages into the project environment."""
    added = sorted(owned_package_dirs(repo) - before)
    passed = not added
    detail = "" if passed else "project environment gained " + ", ".join(added)
    return [
        check_row(
            "project_env_unchanged",
            ["scan", "project-site-packages"],
            exit_code=0 if passed else 1,
            passed=passed,
            detail=detail,
        )
    ]


def module_row(hit: InstallHit) -> dict[str, object]:
    """Check that owned modules resolved from the install."""
    return check_row(
        "modules_from_wheel",
        ["python", "-c", "import fx1, quant_fund, dipcatcher_cli"],
        exit_code=0 if hit.modules_from_install else 1,
        passed=hit.modules_from_install,
        detail=hit.probe_detail,
        import_failure=hit.probe_import_failure,
    )


def _build_failure_row(status: str) -> dict[str, object]:
    return check_row(
        "wheel_build",
        ["uv", "build", "--wheel", "--offline"],
        exit_code=1,
        passed=False,
        detail=f"wheel build {status}",
    )


def choose_keys(keys: tuple[str, str] | None) -> tuple[str, str]:
    """Per-model bearer tokens for the loopback stub. Not a real provider key."""
    if keys is not None:
        return keys
    return ("sk-smoke-fx1-" + secrets.token_hex(8), "sk-smoke-lite-" + secrets.token_hex(8))


def open_stub(repo: Path, keys: Mapping[str, str]) -> _Loopback:
    """Import the repo's loopback stub and construct it."""
    root = str(repo.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("tests.fx1.stub_model_server")
    factory = getattr(module, "StubModelServer", None)
    if not callable(factory):
        raise TypeError("StubModelServer is not callable")
    return cast(_Loopback, factory(dict(keys)))


def start_stub(repo: Path, keys: Mapping[str, str]) -> tuple[_Loopback | None, str, str, bool]:
    """Start the loopback stub. Returns ``(server, url, error, import_failure)``."""
    try:
        server = open_stub(repo, keys)
        server.start()
    except (ImportError, SyntaxError) as exc:
        return None, "", type(exc).__name__, True
    except (OSError, TypeError, RuntimeError) as exc:
        return None, "", type(exc).__name__, False
    return server, server.url, "", False


def close_stub(server: _Loopback | None) -> None:
    """Stop the stub. A close error does not replace the smoke result."""
    if server is None:
        return
    try:
        server.close()
    except (OSError, RuntimeError) as exc:
        print(f"stub close failed: {type(exc).__name__}", file=sys.stderr)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def perform_attempt(
    *,
    repo: Path,
    layout: Layout,
    uv: Path,
    python: Path,
    runner: CommandRunner,
    stub_url: str | None,
    keys: tuple[str, str] | None,
) -> Attempt:
    """Build, install, check, and uninstall once."""
    before = owned_package_dirs(repo)
    attempt = Attempt()
    fx1_key, lite_key = choose_keys(keys)
    secrets = (fx1_key, lite_key)
    wheel, status, mode, notes = build_wheel(uv, repo, layout.wheel, python, runner)
    attempt.notes.extend(notes)
    attempt.build_status = status
    attempt.build_mode = mode
    if wheel is None:
        attempt.checks = [_build_failure_row(status)]
        attempt.checks.extend(leak_rows(repo, before))
        return attempt
    attempt.wheel_path = wheel
    attempt.wheel_name = wheel.name
    attempt.wheel_sha256 = sha256_file(wheel)
    _log(f"built {wheel.name}")
    hit = install_wheel(uv, repo, python, wheel, layout.work, runner, layout.neutral)
    attempt.notes.extend(hit.notes)
    attempt.modules = dict(hit.modules)
    attempt.modules_from_install = hit.modules_from_install
    attempt.install_mode = hit.mode if hit.ok else MODE_NONE
    attempt.checks.append(module_row(hit))
    if not hit.ok or hit.venv is None:
        attempt.checks.extend(_not_installed_rows())
        attempt.checks.extend(leak_rows(repo, before))
        return attempt
    _log(f"install mode {hit.mode}")
    attempt.checks.extend(
        _run_installed_checks(
            hit.venv,
            repo=repo,
            layout=layout,
            uv=uv,
            runner=runner,
            version=wheel_version(wheel.name),
            stub_url=stub_url,
            fx1_key=fx1_key,
            lite_key=lite_key,
            secrets=secrets,
        )
    )
    attempt.checks.extend(leak_rows(repo, before))
    return attempt


def _run_installed_checks(
    venv: Path,
    *,
    repo: Path,
    layout: Layout,
    uv: Path,
    runner: CommandRunner,
    version: str,
    stub_url: str | None,
    fx1_key: str,
    lite_key: str,
    secrets: Sequence[str],
) -> list[dict[str, object]]:
    url, error, imported, server = _resolve_stub(repo, stub_url, fx1_key, lite_key)
    try:
        specs = make_cli_checks(venv, version=version, url=url, fx1_key=fx1_key, lite_key=lite_key)
        rows = run_planned(
            specs,
            runner,
            child_env(layout, venv, {}),
            layout.neutral,
            secrets,
            url_ok=bool(url),
            stub_error=error,
            stub_import=imported,
        )
        rows.extend(removal_checks(venv, uv, runner, layout.neutral, secrets))
        return rows
    finally:
        close_stub(server)


def _resolve_stub(
    repo: Path, stub_url: str | None, fx1_key: str, lite_key: str
) -> tuple[str, str, bool, _Loopback | None]:
    if stub_url:
        return stub_url, "", False, None
    server, url, error, imported = start_stub(repo, {"fx1": fx1_key, "fx1-lite": lite_key})
    return url, error, imported, server


def attempt_passed(attempt: Attempt) -> bool:
    """True only when the build, the install, and every check passed."""
    if attempt.build_status != "ok" or not attempt.modules_from_install:
        return False
    if not attempt.checks:
        return False
    return all(bool(row.get("passed")) for row in attempt.checks)


def assemble_record(
    attempt: Attempt,
    *,
    started_at: str,
    finished_at: str,
    retry: Mapping[str, object],
    uv_version: str,
    base_python: str,
    secrets: Sequence[str],
) -> dict[str, object]:
    """Build the ``dipcatcher.release_smoke.v1`` document."""
    notes = [scrub(bound(note, 500), secrets) for note in attempt.notes]
    return {
        "schema": SCHEMA,
        "passed": attempt_passed(attempt),
        "scope": SCOPE,
        "not_covered": list(NOT_COVERED),
        "strategy": list(STRATEGY),
        "wheel": {"name": attempt.wheel_name, "sha256": attempt.wheel_sha256},
        "build_mode": attempt.build_mode,
        "build_status": attempt.build_status,
        "install_mode": attempt.install_mode,
        "install_notes": notes,
        "modules": dict(attempt.modules),
        "modules_from_install": attempt.modules_from_install,
        "checks": [public_check(row, secrets) for row in attempt.checks],
        "timestamps": {"started_at": started_at, "finished_at": finished_at},
        "platform": platform_facts(),
        "retry": {
            "attempted": bool(retry.get("attempted")),
            "reason": str(retry.get("reason", "")),
        },
        "uv_version": uv_version,
        "base_python": base_python,
        "temporary_venvs_removed": True,
    }


def remove_venvs(work: Path) -> None:
    """Delete install virtualenvs. The wheel directory is left in place."""
    for name in ("venv-pip", "venv-sync", "venv-pth"):
        drop_tree(work / name)


def _now(clock: Callable[[], datetime] | None) -> datetime:
    if clock is None:
        return datetime.now(UTC)
    return clock()


def _uv_version(uv: Path, runner: CommandRunner, cwd: Path) -> str:
    result = run_argv(runner, [str(uv), "--version"], cwd=cwd, env=uv_command_env(), timeout_s=30.0)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def run_release_smoke(
    *,
    repo: Path,
    work_dir: Path,
    out: Path | None,
    checksums: Path | None,
    uv: Path,
    python: Path,
    runner: CommandRunner | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], datetime] | None = None,
    stub_url: str | None = None,
    keys: tuple[str, str] | None = None,
    remove_temporary_venvs: bool = True,
) -> dict[str, object]:
    """Run the smoke and, when requested, write the JSON record and SHA256SUMS."""
    runner_fn = subprocess_runner if runner is None else runner
    sleep_fn = time.sleep if sleep is None else sleep
    layout = make_layout(work_dir)
    started = utc_stamp(_now(clock))
    version = _uv_version(uv, runner_fn, repo)
    chosen, retry, active_keys = _attempts(
        repo=repo,
        layout=layout,
        uv=uv,
        python=python,
        runner=runner_fn,
        stub_url=stub_url,
        keys=keys,
        sleep=sleep_fn,
    )
    if remove_temporary_venvs:
        remove_venvs(work_dir)
    record = assemble_record(
        chosen,
        started_at=started,
        finished_at=utc_stamp(_now(clock)),
        retry=retry,
        uv_version=version,
        base_python=str(python),
        secrets=active_keys,
    )
    _emit(record, out, checksums, chosen.wheel_path, chosen.wheel_name)
    return record


def _attempts(
    *,
    repo: Path,
    layout: Layout,
    uv: Path,
    python: Path,
    runner: CommandRunner,
    stub_url: str | None,
    keys: tuple[str, str] | None,
    sleep: Callable[[float], None],
) -> tuple[Attempt, dict[str, object], tuple[str, str]]:
    active_keys = choose_keys(keys)
    first = perform_attempt(
        repo=repo,
        layout=layout,
        uv=uv,
        python=python,
        runner=runner,
        stub_url=stub_url,
        keys=active_keys,
    )
    reason = import_retry_reason(first.checks)
    if not reason:
        return first, {"attempted": False, "reason": ""}, active_keys
    _log(f"import failure in {reason}; retrying once after {RETRY_DELAY_S:.0f}s")
    sleep(RETRY_DELAY_S)
    second = perform_attempt(
        repo=repo,
        layout=layout,
        uv=uv,
        python=python,
        runner=runner,
        stub_url=stub_url,
        keys=active_keys,
    )
    second.notes.insert(0, f"retried after import failure in {reason}")
    return second, {"attempted": True, "reason": reason}, active_keys


def _emit(
    record: Mapping[str, object],
    out: Path | None,
    checksums: Path | None,
    wheel: Path | None,
    wheel_name: str,
) -> None:
    if out is not None:
        atomic_write_json(out, record)
    else:
        sys.stdout.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if checksums is None:
        return
    artifacts: list[tuple[str, Path]] = []
    if wheel is not None and wheel.is_file() and wheel_name:
        artifacts.append((wheel_name, wheel))
    if out is not None and out.is_file():
        artifacts.append((out.name, out))
    if artifacts:
        write_checksum_file(checksums, artifacts)


def smoke_exit(record: Mapping[str, object]) -> int:
    """Process status: 0 when the record passed, else 1."""
    if record.get("passed") is True:
        return 0
    return 1


def _as_path(value: object) -> Path | None:
    if isinstance(value, Path):
        return value
    return None


def parse_options(argv: Sequence[str]) -> Options:
    """Parse CLI arguments. Paths default to this repo and its virtualenv."""
    parser = argparse.ArgumentParser(
        prog="dipcatcher_release_smoke",
        description=(
            "Build a wheel, install it offline, exercise dipcatcher/quant/fxi, "
            "uninstall, and write a dipcatcher.release_smoke.v1 record. "
            "Signing, rollback drills, and macOS/Windows are not covered."
        ),
    )
    parser.add_argument("--out", type=Path, help="write the JSON record atomically with mode 0600")
    parser.add_argument(
        "--checksums", type=Path, help="write SHA256SUMS for the wheel and JSON record"
    )
    parser.add_argument("--repo", type=Path, help="repository root (default: this script's parent)")
    parser.add_argument(
        "--work-dir", type=Path, help="scratch directory (default: a new temp directory)"
    )
    parser.add_argument("--uv", type=Path, help="uv executable")
    parser.add_argument(
        "--python", type=Path, help="interpreter used to create the install virtualenv"
    )
    parsed = parser.parse_args(list(argv))
    repo = _as_path(parsed.repo) or Path(__file__).resolve().parents[1]
    return Options(
        out=_as_path(parsed.out),
        checksums=_as_path(parsed.checksums),
        repo=repo,
        work_dir=_as_path(parsed.work_dir),
        uv=_as_path(parsed.uv) or default_uv(),
        python=_as_path(parsed.python) or project_python(repo),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: CommandRunner | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], datetime] | None = None,
    stub_url: str | None = None,
    keys: tuple[str, str] | None = None,
) -> int:
    """CLI entry. Returns 0 when every check passed."""
    options = parse_options(sys.argv[1:] if argv is None else argv)
    created = False
    work = options.work_dir
    if work is None:
        work = Path(tempfile.mkdtemp(prefix="dipcatcher-release-smoke-"))
        created = True
    record = run_release_smoke(
        repo=options.repo,
        work_dir=work,
        out=options.out,
        checksums=options.checksums,
        uv=options.uv,
        python=options.python,
        runner=runner,
        sleep=sleep,
        clock=clock,
        stub_url=stub_url,
        keys=keys,
    )
    code = smoke_exit(record)
    _log(
        f"passed={str(record.get('passed')).lower()} "
        f"install_mode={record.get('install_mode')} "
        f"build_status={record.get('build_status')}"
    )
    if created and code == 0:
        shutil.rmtree(work, ignore_errors=True)
    elif created and code != 0:
        _log(f"left scratch directory {work}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
