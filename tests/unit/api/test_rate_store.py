"""Two processes share a sqlite or file rate bucket. Memory stays private."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quant_fund.api.app import app
from quant_fund.api.http_controls import ip_rate_response, reset_rate_gates, take_rate_token

_ROOT = Path(__file__).resolve().parents[3]
_PROBE = """
from quant_fund.api.http_controls import ip_rate_response

class _Client:
    host = "203.0.113.9"

class _Request:
    client = _Client()

response = ip_rate_response(_Request(), "QUANT_API")
print("ok" if response is None else response.status_code)
"""
_HOST = "203.0.113.9"
_KEY = "rate-admin-key"


class _Client:
    def __init__(self, host: str) -> None:
        self.host = host


class _Request:
    def __init__(self, host: str) -> None:
        self.client = _Client(host)


def _child() -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert _KEY not in completed.stdout
    assert _KEY not in completed.stderr
    return completed.stdout.strip()


def _arm(monkeypatch: pytest.MonkeyPatch, backend: str, store: Path) -> None:
    monkeypatch.setenv("DIPCATCHER_RATE_BACKEND", backend)
    monkeypatch.setenv("DIPCATCHER_RATE_STORE", str(store))
    monkeypatch.setenv("QUANT_API_RATE_PER_SEC", "1e-9")
    monkeypatch.setenv("QUANT_API_RATE_BURST", "1")
    monkeypatch.delenv("QUANT_API_LOOPBACK_RATE_PER_SEC", raising=False)
    monkeypatch.delenv("QUANT_API_LOOPBACK_RATE_BURST", raising=False)


@pytest.mark.parametrize("backend", ["sqlite", "file"])
def test_two_replicas_share_one_limit(
    backend: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = tmp_path / ("buckets.sqlite" if backend == "sqlite" else "buckets.json")
    _arm(monkeypatch, backend, store)
    assert ip_rate_response(_Request(_HOST), "QUANT_API") is None
    assert _child() == "429"
    assert store.is_file()
    assert (store.stat().st_mode & 0o777) == 0o600
    assert _KEY.encode() not in store.read_bytes()


def test_memory_replicas_do_not_share_a_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_rate_gates()
    monkeypatch.delenv("DIPCATCHER_RATE_BACKEND", raising=False)
    monkeypatch.delenv("DIPCATCHER_RATE_STORE", raising=False)
    monkeypatch.setenv("QUANT_API_RATE_PER_SEC", "1e-9")
    monkeypatch.setenv("QUANT_API_RATE_BURST", "1")
    assert ip_rate_response(_Request(_HOST), "QUANT_API") is None
    assert _child() == "ok"


def test_a_missing_or_unknown_store_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DIPCATCHER_RATE_BACKEND", "sqlite")
    monkeypatch.delenv("DIPCATCHER_RATE_STORE", raising=False)
    assert take_rate_token("QUANT_API", "QUANT_API:ip:203.0.113.8", 1e-9, 1.0) is None
    monkeypatch.setenv("DIPCATCHER_RATE_BACKEND", "redis")
    monkeypatch.setenv("DIPCATCHER_RATE_STORE", str(tmp_path / "buckets.sqlite"))
    assert take_rate_token("QUANT_API", "QUANT_API:ip:203.0.113.8", 1e-9, 1.0) is None
    blocked = tmp_path / "not-a-file"
    blocked.mkdir()
    monkeypatch.setenv("DIPCATCHER_RATE_BACKEND", "file")
    monkeypatch.setenv("DIPCATCHER_RATE_STORE", str(blocked))
    monkeypatch.setenv("QUANT_API_KEY", _KEY)
    monkeypatch.delenv("QUANT_API_READ_KEY", raising=False)
    client = TestClient(app, client=(_HOST, 9), raise_server_exceptions=False)
    response = client.get("/models", headers={"X-API-Key": _KEY})
    assert response.status_code == 503
    assert response.json() == {"detail": "rate store unavailable"}
    assert _KEY not in response.text


def test_shared_store_keeps_separate_key_buckets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = tmp_path / "buckets.sqlite"
    _arm(monkeypatch, "sqlite", store)
    digest = hashlib.sha256(_KEY.encode()).hexdigest()
    remote = f"QUANT_API:key:remote:{digest}"
    loopback = f"QUANT_API:key:loopback:{digest}"
    assert take_rate_token("QUANT_API", remote, 1e-9, 1.0) is True
    assert take_rate_token("QUANT_API", remote, 1e-9, 1.0) is False
    assert take_rate_token("QUANT_API", loopback, 1e-9, 1.0) is True
    assert _KEY.encode() not in store.read_bytes()
