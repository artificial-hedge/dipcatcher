#!/usr/bin/env python3
"""Deploy N, canary a broken N+1, and measure the automatic rollback.

The broken image is built under ``/tmp`` and exits immediately. It is not
the repo Dockerfile. A failed canary with ``DIPCATCHER_AUTO_ROLLBACK=1``
retags the previous images and recreates the fleet. The seeded ledger
digest must match afterwards. Stdout is one JSON object. API keys are
not printed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_COMPOSE = _ROOT / "deploy" / "compose.prod.yml"
_BROKEN = Path("/tmp/dipcatcher-ops14-broken")
_BROKEN_TAG = "dipcatcher:ops14-broken"
_LEDGER = "broker_state.json"
_SEED = '{"data_label":"SYNTHETIC","schema":"rollback-drill-ledger","step":0}\n'
_RELATIVE = Path("metadata/paper/rollback-drill") / _LEDGER


def _load(name: str, path: Path):
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _docker() -> list[str]:
    sudo = os.environ.get("SUDO")
    if sudo is None:
        sudo = "sudo"
    sudo = sudo.strip()
    if not sudo:
        return ["docker"]
    return [*shlex.split(sudo), "docker"]


def _run(argv: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


def _image_id(name: str) -> str:
    proc = _run([*_docker(), "image", "inspect", "--format", "{{.Id}}", name])
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _container_image(leaf: str) -> str:
    project = os.environ.get("PROJECT", "dipcatcher-ops7").strip() or "dipcatcher-ops7"
    proc = _run([*_docker(), "inspect", "--format", "{{.Image}}", f"{project}-{leaf}-1"])
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _ledger_rows(root: Path) -> list[tuple[str, str]]:
    if not root.is_dir():
        return []
    rows: list[tuple[str, str]] = []
    for path in sorted(root.rglob(_LEDGER)):
        if path.is_symlink() or not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append((path.relative_to(root).as_posix(), digest))
    return rows


def _digest(rows: list[tuple[str, str]]) -> str:
    module = _load("dipcatcher_prod_dr_drill_ops14", _ROOT / "scripts" / "prod_dr_drill.py")
    return str(module.combine_ledger_digest(rows))


def _seed(root: Path) -> bool:
    """Write the synthetic ledger. Lab data is owned by uid 10001."""
    target = root / _RELATIVE
    if target.is_file():
        return True
    sudo = os.environ.get("SUDO")
    if sudo is None:
        sudo = "sudo"
    prefix = shlex.split(sudo.strip()) if sudo.strip() else []
    parent = shlex.quote(str(target.parent))
    dest = shlex.quote(str(target))
    body = shlex.quote(_SEED)
    script = (
        f"mkdir -p {parent} && printf %s {body} > {dest} && "
        f"chown 10001:10001 {parent} {dest} && chmod 755 {parent} && chmod 644 {dest}"
    )
    argv = [*prefix, "sh", "-c", script] if prefix else ["sh", "-c", script]
    proc = _run(argv)
    return proc.returncode == 0 and target.is_file()


def _health() -> bool:
    port = os.environ.get("LAB_TLS_PORT", "8443").strip() or "8443"
    tls = os.environ.get("DIPCATCHER_TLS_DIR", "/tmp/dipcatcher-ops7-tls")
    url = f"https://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=5, cafile=str(Path(tls) / "dev.crt")) as response:
            return int(response.status) == 200
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def _make(target: str, timeout: int) -> int:
    env = os.environ.copy()
    for name in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL"):
        env.pop(name, None)
    proc = subprocess.run(
        ["make", "-f", "deploy/prod.mk", target],
        cwd=_ROOT,
        env=env,
        timeout=timeout,
        check=False,
        stdout=subprocess.DEVNULL,
    )
    return proc.returncode


def _tag(source: str, dest: str) -> bool:
    return _run([*_docker(), "tag", source, dest]).returncode == 0


def _build_broken() -> bool:
    if _BROKEN.exists():
        for child in _BROKEN.iterdir():
            if child.is_file() and not child.is_symlink():
                child.unlink()
    else:
        _BROKEN.mkdir(parents=True)
    (_BROKEN / "Dockerfile").write_text(
        'FROM dipcatcher:local\nENTRYPOINT ["/bin/false"]\n',
        encoding="utf-8",
    )
    proc = _run(
        [*_docker(), "build", "-t", _BROKEN_TAG, "-f", str(_BROKEN / "Dockerfile"), str(_BROKEN)],
        timeout=300,
    )
    return proc.returncode == 0


def _restore_tags(lab_id: str, research_id: str) -> None:
    if lab_id:
        _tag(lab_id, "dipcatcher:local")
    if research_id:
        _tag(research_id, "dipcatcher-research-api:latest")
    _run([*_docker(), "rmi", _BROKEN_TAG])


def _fleet_images() -> dict[str, str]:
    return {leaf: _container_image(leaf) for leaf in ("lab-a", "lab-b", "research-a", "research-b")}


def execute() -> dict[str, object]:
    """Bring N up when needed, fail the canary on N+1, and roll back to N."""
    if (
        not os.environ.get("QUANT_API_KEY", "").strip()
        or not os.environ.get("RESEARCH_API_KEY", "").strip()
    ):
        return {"error": "keys", "ok": False}
    started_here = False
    lab_id = ""
    research_id = ""
    try:
        if not _health():
            if _make("prod-up", 900) != 0:
                return {"error": "prod-up", "ok": False}
            started_here = True
        lab_data = Path(os.environ.get("DIPCATCHER_LAB_DATA", "/tmp/dipcatcher-ops7-data/lab"))
        if not _seed(lab_data):
            return {"error": "seed", "ok": False}
        before_rows = _ledger_rows(lab_data)
        before = _digest(before_rows)
        lab_id = _image_id("dipcatcher:local")
        research_id = _image_id("dipcatcher-research-api:latest")
        before_fleet = _fleet_images()
        if (
            not before
            or not lab_id
            or not research_id
            or any(not item for item in before_fleet.values())
        ):
            return {"error": "baseline", "ok": False}
        if not _tag(lab_id, "dipcatcher:previous") or not _tag(
            research_id, "dipcatcher-research-api:previous"
        ):
            return {"error": "tag-previous", "ok": False}
        if not _build_broken() or not _tag(_BROKEN_TAG, "dipcatcher:local"):
            return {"error": "broken-image", "ok": False}
        if not _tag(_BROKEN_TAG, "dipcatcher-research-api:latest"):
            return {"error": "broken-research", "ok": False}
        canary = _load("dipcatcher_prod_canary_ops14", _ROOT / "scripts" / "prod_canary.py")
        env = dict(os.environ)
        env["DIPCATCHER_AUTO_ROLLBACK"] = "1"
        notes: list[str] = []

        def _run_canary(argv: list[str]) -> subprocess.CompletedProcess[str]:
            proc = _run(list(argv), timeout=400)
            if proc.returncode != 0 and len(notes) < 4:
                err = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")
                notes.append(err[:240])
            return proc

        report = canary.execute(env, _run_canary, attempts=12)
        health_started = time.monotonic()
        gate = _run([sys.executable, str(_ROOT / "scripts" / "prod_rollback.py")], timeout=200)
        health_s = time.monotonic() - health_started
        rollback_s = float(report.get("rollback_seconds") or 0.0) + health_s
        after_rows = _ledger_rows(lab_data)
        after = _digest(after_rows)
        after_fleet = _fleet_images()
        restored = (
            _image_id("dipcatcher:local") == lab_id
            and _image_id("dipcatcher-research-api:latest") == research_id
        )
        fleet_ok = after_fleet == before_fleet
        ok = (
            report.get("action") == "abort"
            and report.get("rollback_ok") is True
            and gate.returncode == 0
            and after == before
            and before_rows == after_rows
            and restored
            and fleet_ok
        )
        canary_error = str(report.get("error") or "")
        return {
            "action": report.get("action"),
            "command_errors": notes,
            "error": "" if ok else (canary_error or "rollback"),
            "fleet_restored": fleet_ok,
            "ledger_digest": after,
            "ledger_unchanged": after == before and before_rows == after_rows,
            "ok": ok,
            "rollback_ok": report.get("rollback_ok"),
            "tags_restored": restored,
            "time_to_rollback_seconds": round(rollback_s, 3),
        }
    finally:
        _restore_tags(lab_id, research_id)
        if started_here:
            _make("prod-down", 180)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-rollback-drill takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
