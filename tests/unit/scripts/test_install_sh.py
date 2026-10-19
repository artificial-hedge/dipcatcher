"""install.sh supply-chain gates, exercised by running the real installer.

A remote (http/https) wheel installs only under a matching
``FXI_WHEEL_SHA256``; ``--requirements FILE`` / ``FXI_REQUIREMENTS`` locks
the dependency set (``--require-hashes``, wheel added ``--no-deps``,
``pip check`` for anything the file left out). The installer runs under
POSIX ``sh`` against a localhost ``http.server`` serving stub wheels. pip runs
with ``--no-index`` plus ``--find-links`` on that wheelhouse, and every proxy
variable points at a dead port, so nothing leaves the machine.
"""

from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import sys
import threading
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="install.sh is a POSIX sh installer"
)

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = REPO_ROOT / "install.sh"
VERSION = "0.0.1"
WHEEL = f"fx_1-{VERSION}-py3-none-any.whl"
DEAD_PROXY = "http://127.0.0.1:9"
# The installer is `#!/bin/sh` and piped to `sh`; bash runs it too.
SHELLS = ["sh", "bash"]


def _record_hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + digest.decode()


def _build_wheel(
    out: Path,
    name: str,
    version: str,
    files: dict[str, str],
    requires: tuple[str, ...] = (),
    scripts: dict[str, str] | None = None,
) -> Path:
    dist_info = f"{name}-{version}.dist-info"
    entries = dict(files)
    entries[f"{dist_info}/METADATA"] = (
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        + "".join(f"Requires-Dist: {req}\n" for req in requires)
    )
    entries[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    if scripts:
        entries[f"{dist_info}/entry_points.txt"] = "[console_scripts]\n" + "".join(
            f"{cmd} = {target}\n" for cmd, target in scripts.items()
        )
    record = [f"{p},{_record_hash(c.encode())},{len(c.encode())}" for p, c in entries.items()]
    entries[f"{dist_info}/RECORD"] = "\n".join([*record, f"{dist_info}/RECORD,,"]) + "\n"
    path = out / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for member, content in entries.items():
            archive.writestr(member, content)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Wheelhouse:
    root: Path
    base_url: str
    requested: list[str]

    def url(self, name: str) -> str:
        return f"{self.base_url}/{name}"

    def pin(self, name: str) -> str:
        """A hashed ``name==1.0`` requirement line for a stub dependency wheel."""
        digest = _sha256(self.root / f"{name}-1.0-py3-none-any.whl")
        return f"{name}==1.0 --hash=sha256:{digest}"


class _RecordingHandler(SimpleHTTPRequestHandler):
    requested: list[str]

    def do_GET(self) -> None:
        self.requested.append(self.path)
        super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return


@pytest.fixture(scope="module")
def wheelhouse(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Wheelhouse]:
    root = tmp_path_factory.mktemp("wheelhouse")
    _build_wheel(root, "fxdep", "1.0", {"fxdep.py": "X = 1\n"})
    _build_wheel(root, "fxother", "1.0", {"fxother.py": "Y = 2\n"})
    # fxdep is declared but not imported: only `pip check` can notice it missing.
    _build_wheel(
        root,
        "fx_1",
        VERSION,
        {"fxi_stub.py": f"def main():\n    print('fxi {VERSION}')\n"},
        requires=("fxdep==1.0",),
        scripts={"fxi": "fxi_stub:main"},
    )
    requested: list[str] = []
    handler = type("Handler", (_RecordingHandler,), {"requested": requested})
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(handler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Wheelhouse(root, f"http://127.0.0.1:{server.server_address[1]}", requested)
    finally:
        server.shutdown()
        server.server_close()


def _run(
    tmp_path: Path, house: Wheelhouse, *args: str, shell: str = "sh", **env_extra: str
) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "FXI_PYTHON": sys.executable,
        "FXI_VERSION": VERSION,
        "FXI_HOME": str(tmp_path / "fxi"),
        "FXI_BIN_DIR": str(tmp_path / "bin"),
        "PIP_NO_INDEX": "1",
        "PIP_FIND_LINKS": str(house.root),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_RETRIES": "0",
        "http_proxy": DEAD_PROXY,
        "https_proxy": DEAD_PROXY,
        "HTTP_PROXY": DEAD_PROXY,
        "HTTPS_PROXY": DEAD_PROXY,
        "no_proxy": "127.0.0.1,localhost",
        "NO_PROXY": "127.0.0.1,localhost",
        **env_extra,
    }
    return subprocess.run(
        [shell, str(INSTALL_SH), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def _installed(tmp_path: Path) -> bool:
    return (tmp_path / "bin" / "fxi").exists()


def _venv_has(tmp_path: Path, module: str) -> bool:
    probe = subprocess.run(
        [str(tmp_path / "fxi" / "venv" / "bin" / "python"), "-c", f"import {module}"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def _requirements(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "requirements.lock.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("scheme", ["http", "https"])
def test_remote_wheel_without_sha256_pin_is_refused(
    tmp_path: Path, wheelhouse: Wheelhouse, scheme: str, shell: str
) -> None:
    before = len(wheelhouse.requested)
    url = wheelhouse.url(WHEEL) if scheme == "http" else f"https://127.0.0.1:1/{WHEEL}"
    proc = _run(tmp_path, wheelhouse, shell=shell, FXI_INSTALL_URL=url)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "FXI_WHEEL_SHA256" in proc.stderr
    assert "refusing" in proc.stderr
    # Refused before any download or write: no fetch, no FXI_HOME, no links.
    assert len(wheelhouse.requested) == before
    assert not (tmp_path / "fxi").exists()
    assert not _installed(tmp_path)


def test_remote_wheel_sha256_mismatch_is_refused(tmp_path: Path, wheelhouse: Wheelhouse) -> None:
    proc = _run(
        tmp_path, wheelhouse, FXI_INSTALL_URL=wheelhouse.url(WHEEL), FXI_WHEEL_SHA256="0" * 64
    )
    assert proc.returncode != 0
    assert "sha256 mismatch" in proc.stderr
    assert not (tmp_path / "fxi" / "venv").exists()
    assert not _installed(tmp_path)


def test_pinned_remote_wheel_installs_and_warns_unlocked(
    tmp_path: Path, wheelhouse: Wheelhouse
) -> None:
    proc = _run(
        tmp_path,
        wheelhouse,
        FXI_INSTALL_URL=wheelhouse.url(WHEEL),
        FXI_WHEEL_SHA256=_sha256(wheelhouse.root / WHEEL),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "sha256 verified" in proc.stdout
    assert "--requirements" in proc.stderr  # unlocked dependency resolution is warned
    assert _installed(tmp_path)
    smoke = subprocess.run([str(tmp_path / "bin" / "fxi")], capture_output=True, text=True)
    assert smoke.stdout.strip() == f"fxi {VERSION}"


@pytest.mark.parametrize("shell", SHELLS)
def test_hashed_requirements_lock_the_dependency_set(
    tmp_path: Path, wheelhouse: Wheelhouse, shell: str
) -> None:
    reqs = _requirements(tmp_path, wheelhouse.pin("fxdep"))
    proc = _run(
        tmp_path,
        wheelhouse,
        "--requirements",
        str(reqs),
        shell=shell,
        FXI_INSTALL_URL=wheelhouse.url(WHEEL),
        FXI_WHEEL_SHA256=_sha256(wheelhouse.root / WHEEL),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--requirements" not in proc.stderr  # locked: no unlocked warning
    assert _installed(tmp_path)
    assert _venv_has(tmp_path, "fxdep")


def test_dependency_missing_from_requirements_fails_closed(
    tmp_path: Path, wheelhouse: Wheelhouse
) -> None:
    # fxdep is findable on the wheelhouse, so only --no-deps keeps pip from
    # resolving it outside the lock; pip check then refuses the gap.
    reqs = _requirements(tmp_path, wheelhouse.pin("fxother"))
    proc = _run(
        tmp_path, wheelhouse, "--wheel", str(wheelhouse.root / WHEEL), FXI_REQUIREMENTS=str(reqs)
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "fxdep" in proc.stdout + proc.stderr
    assert "pip check" in proc.stderr
    assert not (tmp_path / "fxi" / "venv").exists()
    assert not _installed(tmp_path)


@pytest.mark.parametrize("line", ["fxdep==1.0", "fxdep==1.0 --hash=sha256:" + "0" * 64])
def test_requirements_entries_must_carry_matching_hashes(
    tmp_path: Path, wheelhouse: Wheelhouse, line: str
) -> None:
    reqs = _requirements(tmp_path, line)
    proc = _run(
        tmp_path,
        wheelhouse,
        FXI_INSTALL_URL=wheelhouse.url(WHEEL),
        FXI_WHEEL_SHA256=_sha256(wheelhouse.root / WHEEL),
        FXI_REQUIREMENTS=str(reqs),
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "hash" in proc.stderr.lower()
    assert not _installed(tmp_path)


@pytest.mark.parametrize("where", ["missing", "url"])
def test_requirements_must_be_a_local_file(
    tmp_path: Path, wheelhouse: Wheelhouse, where: str
) -> None:
    source = str(tmp_path / "absent.txt") if where == "missing" else wheelhouse.url("reqs.txt")
    proc = _run(
        tmp_path,
        wheelhouse,
        FXI_INSTALL_URL=wheelhouse.url(WHEEL),
        FXI_WHEEL_SHA256=_sha256(wheelhouse.root / WHEEL),
        FXI_REQUIREMENTS=source,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "requirements file not found" in proc.stderr
    assert not (tmp_path / "fxi").exists()
    assert not _installed(tmp_path)


def test_locked_install_does_not_inherit_a_reused_venv(
    tmp_path: Path, wheelhouse: Wheelhouse
) -> None:
    pin = _sha256(wheelhouse.root / WHEEL)
    first = _run(tmp_path, wheelhouse, FXI_INSTALL_URL=wheelhouse.url(WHEEL), FXI_WHEEL_SHA256=pin)
    assert first.returncode == 0, first.stdout + first.stderr
    assert _venv_has(tmp_path, "fxdep")  # resolved from the index, unlocked
    # The lock omits fxdep: the unhashed copy left by the first run must not
    # satisfy it.
    reqs = _requirements(tmp_path, wheelhouse.pin("fxother"))
    second = _run(
        tmp_path,
        wheelhouse,
        FXI_INSTALL_URL=wheelhouse.url(WHEEL),
        FXI_WHEEL_SHA256=pin,
        FXI_REQUIREMENTS=str(reqs),
    )
    assert second.returncode != 0, second.stdout + second.stderr
    assert "pip check" in second.stderr
    assert not _installed(tmp_path)
