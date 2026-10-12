"""Golden subprocess contract for ``dipcatcher models``."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fx1.stub_model_server import StubModelServer
from tests.fx1.ux_pty import hermetic_network

ROOT = Path(__file__).resolve().parents[3]
_PY = sys.executable
_KEY = "sk-models-contract-secret-aaa"
_LIST_KEYS = {"schema", "models", "store_warnings"}
_ROW_KEYS = {"model", "host", "status", "source", "fingerprint", "set_at"}
_CHECK_KEYS = {
    "schema",
    "live",
    "generated_at",
    "fx1_version",
    "contacted",
    "note",
    "store_warnings",
    "models",
}
_CHECK_ROW_KEYS = {
    "requested_model",
    "served_model",
    "matched",
    "request_id",
    "host",
    "latency_ms",
    "usage",
    "status",
    "error_kind",
    "hint",
    "configured",
    "presence",
    "source",
    "fingerprint",
    "url_valid",
    "contacted",
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
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
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


def test_dipcatcher_contract_models_list_json(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env["MOONSHOT_API_KEY"] = _KEY
    proc = _run(["models", "list", "--json"], env)
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert set(payload) == _LIST_KEYS
    assert payload["schema"] == "dipcatcher.models.v1"
    assert [row["model"] for row in payload["models"]] == ["fx1", "fx1-lite"]
    assert set(payload["models"][0]) == _ROW_KEYS
    assert _KEY not in proc.stdout + proc.stderr
    assert "available" not in proc.stdout.casefold()


def test_dipcatcher_contract_models_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    missing = _run(["models", "check", "--json"], env)
    assert missing.returncode == 5
    body = json.loads(missing.stdout)
    assert set(body) == _CHECK_KEYS
    assert body["schema"] == "dipcatcher.models_check.v1"
    assert body["contacted"] is False
    assert set(body["models"][0]) == _CHECK_ROW_KEYS

    unknown = _run(["models", "check", "other"], env)
    assert unknown.returncode == 2

    server = StubModelServer({"fx1": _KEY, "fx1-lite": _KEY})
    server.start()
    try:
        from fx1.interactive.profiles import set_endpoint

        set_endpoint("fx1", _KEY, server.url)
        set_endpoint("fx1-lite", _KEY, server.url)
        offline = _run(["models", "check"], env)
        assert offline.returncode == 0
        assert "nothing was contacted" in offline.stdout
        assert server.requests == []
        out = tmp_path / "check.json"
        live = _run(["models", "check", "--live", "--json", "--out", str(out)], env)
        assert live.returncode == 0
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved["contacted"] is True
        assert stat.S_IMODE(out.stat().st_mode) == 0o600
        assert _KEY not in live.stdout + live.stderr + out.read_text(encoding="utf-8")
    finally:
        server.close()
