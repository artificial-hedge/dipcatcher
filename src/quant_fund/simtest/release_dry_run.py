"""Build the wheel and sdist, then run the installed CLI entrypoints.

Mirrors the CI package job locally: ``uv build`` offline, install that wheel
with ``--no-deps`` into a prefix, and require ``dipcatcher --help`` and
``fx1 --help`` to exit 0 with a Usage line. The project environment is left
on the checkout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_ENTRYPOINTS = ("dipcatcher", "fx1")
_TIMEOUT_S = 180


class ReleaseDryRunError(RuntimeError):
    """The built artifact did not produce a working CLI entrypoint."""


def run_release_dry_run(work: str | Path, *, repo: str | Path | None = None) -> dict[str, Any]:
    """Build into ``work`` and return wheel, sdist, and help sizes."""
    root = Path(repo) if repo is not None else _repo_root()
    dest = Path(work)
    dist = dest / "dist"
    prefix = dest / "prefix"
    _build(root, dist)
    wheel, sdist = _artifacts(dist)
    site = _install(wheel, prefix)
    _prove_artifact(site)
    helps = {name: _entrypoint_help(prefix, site, name) for name in _ENTRYPOINTS}
    return {"wheel": wheel.name, "sdist": sdist.name, "entrypoints": helps}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build(repo: Path, dist: Path) -> None:
    dist.mkdir(parents=True, exist_ok=True)
    _run(
        ["uv", "build", "--offline", "--out-dir", str(dist), "--no-create-gitignore"],
        cwd=repo,
    )


def _artifacts(dist: Path) -> tuple[Path, Path]:
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseDryRunError("release dry-run expected one wheel and one sdist")
    if wheels[0].stat().st_size == 0 or sdists[0].stat().st_size == 0:
        raise ReleaseDryRunError("release dry-run built an empty artifact")
    return wheels[0], sdists[0]


def _install(wheel: Path, prefix: Path) -> Path:
    prefix.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "uv",
            "pip",
            "install",
            "--offline",
            "--python",
            sys.executable,
            "--no-deps",
            "--prefix",
            str(prefix),
            str(wheel),
        ],
        cwd=prefix,
    )
    sites = sorted(prefix.glob("lib/python*/site-packages"))
    if len(sites) != 1:
        raise ReleaseDryRunError("release dry-run install missed site-packages")
    return sites[0]


def _prove_artifact(site: Path) -> None:
    code = (
        "import dipcatcher_cli, fx1, quant_fund; "
        "print(dipcatcher_cli.__file__); print(fx1.__file__); print(quant_fund.__file__)"
    )
    proc = _spawn([sys.executable, "-c", code], cwd=site, env=_help_env(site))
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or len(lines) != 3:
        raise ReleaseDryRunError("release dry-run could not import the wheel")
    if any(not line.startswith(str(site)) for line in lines):
        raise ReleaseDryRunError("release dry-run imported a tree other than the wheel")


def _entrypoint_help(prefix: Path, site: Path, name: str) -> dict[str, int]:
    script = prefix / "bin" / name
    if not script.is_file() or script.stat().st_size == 0:
        raise ReleaseDryRunError(f"release dry-run missing entrypoint {name}")
    proc = _spawn([str(script), "--help"], cwd=prefix, env=_help_env(site))
    text = proc.stdout
    if proc.returncode != 0 or not text.strip() or "Usage" not in text:
        raise ReleaseDryRunError(f"release dry-run {name} --help failed")
    return {"exit_code": proc.returncode, "help_bytes": len(text.encode("utf-8"))}


def _help_env(site: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(site)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    return env


def _run(argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    proc = _spawn(argv, cwd=cwd, env=None)
    if proc.returncode != 0:
        raise ReleaseDryRunError(_short(proc))
    return proc


def _spawn(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        check=False,
    )


def _short(proc: subprocess.CompletedProcess[str]) -> str:
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    return "release dry-run command failed: " + " | ".join(err[-4:])[:400]
