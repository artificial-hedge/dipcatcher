"""Golden subprocess contract for ``dipcatcher sessions`` and ``resume``."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from tests.fx1.ux_pty import hermetic_network

ROOT = Path(__file__).resolve().parents[3]
_PY = sys.executable
_SECRET = "sk-session-contract-secret"
_LIST_KEYS = {"schema", "command", "sessions", "warnings"}
_ROW_KEYS = {"id", "model", "status", "turns", "updated"}
_SHOW_KEYS = {
    "id",
    "model",
    "status",
    "revision",
    "created_at",
    "updated_at",
    "truncated",
    "job",
    "turns",
}
_PRUNE_KEYS = {
    "schema",
    "command",
    "dry_run",
    "deleted",
    "planned",
    "max_sessions",
    "max_age_days",
    "warnings",
}


def _env(tmp_path: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.casefold().endswith(("_api_key", "_token", "_secret"))
    }
    for name in ("cfg", "sessions", "home"):
        (tmp_path / name).mkdir(exist_ok=True)
    env.update(
        {
            "FX1_CONFIG_DIR": str(tmp_path / "cfg"),
            "FX1_SESSION_DIR": str(tmp_path / "sessions"),
            "HOME": str(tmp_path / "home"),
            "NO_COLOR": "1",
            "TERM": "dumb",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    env.update(hermetic_network())
    return env


def _run(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_PY, "-m", "dipcatcher_cli", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _seed(tmp_path: Path) -> str:
    from fx1.interactive.sessions import SessionStore

    store = SessionStore(tmp_path / "sessions")
    record = store.create(
        model="fx1",
        messages=[{"role": "user", "content": f"hello {_SECRET}"}],
    )
    assert isinstance(record, dict)
    return str(record["id"])


def test_dipcatcher_contract_sessions_round_trip(tmp_path: Path) -> None:
    env = _env(tmp_path)
    session_id = _seed(tmp_path)
    listed = _run(["sessions", "list", "--json"], env)
    assert listed.returncode == 0
    payload = json.loads(listed.stdout)
    assert set(payload) == _LIST_KEYS
    assert payload["schema"] == "dipcatcher.sessions.v1"
    assert payload["command"] == "list"
    assert set(payload["sessions"][0]) == _ROW_KEYS
    assert _SECRET not in listed.stdout + listed.stderr

    shown = _run(["sessions", "show", session_id, "--json", "--messages"], env)
    assert shown.returncode == 0
    view = json.loads(shown.stdout)["session"]
    assert set(view) >= _SHOW_KEYS
    assert _SECRET not in shown.stdout

    exported = _run(["sessions", "export", session_id, "--format", "md"], env)
    assert exported.returncode == 0
    assert _SECRET not in exported.stdout
    out = tmp_path / "export.md"
    filed = _run(["sessions", "export", "last", "--out", str(out)], env)
    assert filed.returncode == 0
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert _SECRET not in out.read_text(encoding="utf-8")
    again = _run(["sessions", "export", "last", "--out", str(out)], env)
    assert again.returncode == 2

    pruned = _run(["sessions", "prune", "--max-sessions", "0", "--dry-run", "--json"], env)
    assert pruned.returncode == 0
    prune_body = json.loads(pruned.stdout)
    assert set(prune_body) == _PRUNE_KEYS
    assert session_id in prune_body["planned"]

    removed = _run(["sessions", "rm", session_id, "--yes"], env)
    assert removed.returncode == 0
    missing = _run(["sessions", "show", session_id], env)
    assert missing.returncode == 2
    assert _SECRET not in missing.stderr


def test_dipcatcher_contract_resume_needs_a_tty(tmp_path: Path) -> None:
    env = _env(tmp_path)
    proc = _run(["resume"], env)
    assert proc.returncode == 2
    assert "interactive terminal" in proc.stderr
    help_text = _run(["sessions", "--help"], env)
    assert help_text.returncode == 0
    for name in ("list", "show", "export", "rm", "prune"):
        assert name in help_text.stdout
