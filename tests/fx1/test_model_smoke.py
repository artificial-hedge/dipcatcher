"""Subprocess evidence runs of scripts/fx1_model_smoke.py against the loopback stub.

The canonical content hash is independent of the script: hex SHA-256 of the
UTF-8 JSON object with ``content_sha256`` removed, serialized with
``json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True)``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import fx1
from fx1.interactive.profiles import set_endpoint
from tests.fx1.stub_model_server import Reply, StubModelServer

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "fx1_model_smoke.py"
_SECRET = "smoke-secret-fx1-ZZ99qq"
_KEY_ENV = "SMOKE_API_KEY"
_STORED_FX1 = "stored-fx1-secret-AAAA"
_STORED_LITE = "stored-lite-secret-BBBB"
_DECOY_SHARED = "decoy-shared-secret-CCCC"
_DECOY_MODEL = "decoy-model-secret-DDDD"
_PROMPT = "Reply with the single word pong."
_ROW_KEYS = {
    "completion_latency_ms",
    "completion_ok",
    "endpoint_host",
    "error_kind",
    "failed_stage",
    "hint",
    "identity_match",
    "latency_ms",
    "ok",
    "probe_latency_ms",
    "probe_ok",
    "probe_request_id",
    "probe_usage",
    "request_id",
    "requested_model",
    "response_id",
    "retry_after_s",
    "served_model",
    "status",
    "summary",
    "system_fingerprint",
    "usage",
}
_TOP_KEYS = {
    "completion_request",
    "content_sha256",
    "disclaimer",
    "does_not_establish",
    "finished_at",
    "generated_at",
    "git_revision",
    "limitations",
    "models",
    "ok",
    "partial",
    "policy",
    "probe_request",
    "schema",
    "started_at",
    "tool",
    "tool_version",
}


def _child_env(tmp: Path, **extra: str) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp / "home"),
        "FX1_CONFIG_DIR": str(tmp / "cfg"),
        "FX1_SESSION_DIR": str(tmp / "sessions"),
        "PYTHONPATH": str(_ROOT / "src"),
        "NO_COLOR": "1",
        "TERM": "dumb",
        "FORCE_COLOR": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONFAULTHANDLER": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    for name in ("SYSTEMROOT", "COMSPEC", "PATHEXT"):
        value = os.environ.get(name)
        if value:
            env[name] = value
    env.update(extra)
    return env


def _run(
    tmp: Path,
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 8.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *argv],
        cwd=cwd or tmp,
        env=env if env is not None else _child_env(tmp),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _content_sha256(document: dict[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "content_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_evidence(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    loaded = json.loads(raw)
    pretty = json.dumps(loaded, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    assert raw == pretty
    assert isinstance(loaded, dict)
    body = dict(loaded)
    digest = body.pop("content_sha256")
    assert digest == _content_sha256(body)
    assert isinstance(digest, str) and len(digest) == 64
    loaded["content_sha256"] = digest
    return loaded


def _blob(proc: subprocess.CompletedProcess[str], path: Path | None) -> str:
    file_text = ""
    if path is not None and path.is_file():
        file_text = path.read_text(encoding="utf-8", errors="replace")
    return proc.stdout + proc.stderr + file_text


def _assert_secret_absent(blob: str, *secrets: str) -> None:
    for secret in secrets:
        assert secret not in blob
        assert f"Bearer {secret}" not in blob
        assert f"user:{secret}" not in blob
        assert f"api_key={secret}" not in blob


def _assert_concrete_url_absent(blob: str, *, url: str, base_url: str, port: int) -> None:
    assert url not in blob
    assert base_url not in blob
    assert f":{port}" not in blob


def _base_argv(server: StubModelServer, out: Path, *extra: str) -> list[str]:
    return [
        "--base-url",
        server.url,
        "--api-key-env",
        _KEY_ENV,
        "--timeout-s",
        "2",
        "--out",
        str(out),
        *extra,
    ]


def _git_head() -> str:
    proc = subprocess.run(
        ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=2,
    )
    return proc.stdout.strip()


def _assert_contract(document: dict[str, Any]) -> None:
    assert set(document) == _TOP_KEYS
    assert list(document) == sorted(document)
    assert document["schema"] == "model_smoke.v1"
    assert document["tool"] == "fx1_model_smoke"
    assert document["tool_version"] == fx1.__version__
    assert document["git_revision"] == _git_head()
    assert document["does_not_establish"] == [
        "model release status",
        "quality",
        "availability for anyone else",
    ]
    disclaimer = document["disclaimer"]
    assert isinstance(disclaimer, str)
    assert "proves the harness works with the supplied endpoint" in disclaimer
    assert "does not establish model release status" in disclaimer
    assert document["generated_at"][:10] in disclaimer
    assert document["started_at"] <= document["finished_at"]
    assert document["generated_at"] == document["finished_at"]
    for stamp in (document["started_at"], document["finished_at"]):
        parsed = datetime.fromisoformat(stamp)
        assert parsed.tzinfo is not None
    assert document["limitations"] == []
    assert document["probe_request"] == {
        "max_tokens": 1,
        "method": "ModelClient.ping",
        "prompt": "ping",
    }
    assert document["completion_request"] == {
        "max_tokens": 16,
        "prompt": _PROMPT,
        "seed": 0,
        "temperature": 0.0,
    }


def _assert_row_shape(row: dict[str, Any]) -> None:
    assert set(row) == _ROW_KEYS
    assert list(row) == sorted(row)
    response_id = row["response_id"]
    assert response_id is None or isinstance(response_id, str)
    for key in ("latency_ms", "probe_latency_ms", "completion_latency_ms"):
        assert isinstance(row[key], (int, float))
        assert row[key] >= 0


def test_both_models_success_evidence(tmp_path: Path) -> None:
    out = tmp_path / "evidence.json"
    with StubModelServer(keys={"fx1": _SECRET, "fx1-lite": _SECRET}) as server:
        argv = _base_argv(server, out)
        assert _SECRET not in argv
        proc = _run(tmp_path, argv, env=_child_env(tmp_path, **{_KEY_ENV: _SECRET}))
        requests = server.requests
        endpoint = (server.url, server.base_url, server.port)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert proc.stderr == ""
    document = _load_evidence(out)
    _assert_contract(document)
    assert document["ok"] is True
    assert document["partial"] is False
    assert document["policy"] == "require-both"
    assert [row["requested_model"] for row in document["models"]] == ["fx1", "fx1-lite"]
    blob = _blob(proc, out)
    _assert_secret_absent(blob, _SECRET)
    _assert_concrete_url_absent(blob, url=endpoint[0], base_url=endpoint[1], port=endpoint[2])
    assert "/v1/chat/completions" not in out.read_text(encoding="utf-8")
    assert "http://" not in out.read_text(encoding="utf-8")
    assert document["disclaimer"] in proc.stdout
    assert "not evidence for both models" not in proc.stdout
    assert list(out.parent.glob("*.tmp")) == []
    for row, probe_id, request_id in (
        (document["models"][0], "stub-1", "stub-2"),
        (document["models"][1], "stub-3", "stub-4"),
    ):
        _assert_row_shape(row)
        assert row["ok"] is True
        assert row["probe_ok"] is True
        assert row["completion_ok"] is True
        assert row["failed_stage"] is None
        assert row["error_kind"] is None
        assert row["hint"] is None
        assert row["identity_match"] == "match"
        assert row["served_model"] == row["requested_model"]
        assert row["endpoint_host"] == "127.0.0.1"
        assert row["probe_request_id"] == probe_id
        assert row["request_id"] == request_id
        assert row["response_id"] == "chatcmpl-stub"
        assert row["system_fingerprint"] == "stub"
        assert row["status"] is None
        assert row["usage"] == {"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2}
        assert row["probe_usage"] == row["usage"]
        assert f"{row['requested_model']} ok host=127.0.0.1 identity=match" in proc.stdout
    assert [item.model for item in requests] == ["fx1", "fx1", "fx1-lite", "fx1-lite"]
    assert all(item.path.endswith("/v1/chat/completions") for item in requests)
    assert all(item.auth_ok for item in requests)
    ping, completion = requests[0], requests[1]
    assert ping.body is not None and completion.body is not None
    assert ping.body["messages"] == [{"role": "user", "content": "ping"}]
    assert ping.body["max_tokens"] == 1
    assert completion.body["messages"] == [{"role": "user", "content": _PROMPT}]
    assert completion.body["max_tokens"] == 16
    assert completion.body["seed"] == 0
    assert completion.body["temperature"] == 0


def test_default_output_creates_data_fx1(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    assert not (work / "data").exists()
    with StubModelServer(keys={"fx1": _SECRET, "fx1-lite": _SECRET}) as server:
        argv = [
            "--base-url",
            server.url,
            "--api-key-env",
            _KEY_ENV,
            "--timeout-s",
            "2",
        ]
        proc = _run(tmp_path, argv, cwd=work, env=_child_env(tmp_path, **{_KEY_ENV: _SECRET}))
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    found = list((work / "data" / "fx1").glob("model_smoke_*.json"))
    assert len(found) == 1
    assert re.fullmatch(r"model_smoke_\d{8}T\d{6}Z\.json", found[0].name)
    document = _load_evidence(found[0])
    assert document["ok"] is True
    assert document["partial"] is False
    assert str(found[0].resolve()) in proc.stdout


def test_out_outside_repo(tmp_path: Path) -> None:
    outside = tmp_path / "nested" / "evidence.json"
    assert not outside.resolve().is_relative_to(_ROOT)
    with StubModelServer(keys={"fx1": _SECRET, "fx1-lite": _SECRET}) as server:
        argv = _base_argv(server, outside)
        proc = _run(tmp_path, argv, env=_child_env(tmp_path, **{_KEY_ENV: _SECRET}))
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert outside.is_file()
    document = _load_evidence(outside)
    assert document["ok"] is True
    _assert_secret_absent(_blob(proc, outside), _SECRET)


def test_allow_partial_records_one_model(tmp_path: Path) -> None:
    out = tmp_path / "partial.json"
    with StubModelServer(keys={"fx1": _SECRET, "fx1-lite": _SECRET}) as server:
        argv = _base_argv(server, out, "--allow-partial", "--model", "fx1-lite")
        proc = _run(tmp_path, argv, env=_child_env(tmp_path, **{_KEY_ENV: _SECRET}))
        models = [item.model for item in server.requests]
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    document = _load_evidence(out)
    assert document["partial"] is True
    assert document["policy"] == "allow-partial"
    assert document["ok"] is True
    assert [row["requested_model"] for row in document["models"]] == ["fx1-lite"]
    assert models == ["fx1-lite", "fx1-lite"]
    assert "partial: requested fx1-lite; this is not evidence for both models" in proc.stdout
    assert document["disclaimer"] in proc.stdout


def test_require_both_fails_closed_when_one_model_is_unknown(tmp_path: Path) -> None:
    out = tmp_path / "both.json"
    with StubModelServer(keys={"fx1": _SECRET}) as server:
        argv = _base_argv(server, out)
        proc = _run(tmp_path, argv, env=_child_env(tmp_path, **{_KEY_ENV: _SECRET}))
        count = len(server.requests)
    assert proc.returncode == 1, f"{proc.stdout}\n{proc.stderr}"
    document = _load_evidence(out)
    assert document["ok"] is False
    assert document["partial"] is False
    assert document["policy"] == "require-both"
    fx1, lite = document["models"]
    assert fx1["ok"] is True and fx1["requested_model"] == "fx1"
    assert lite["requested_model"] == "fx1-lite"
    assert lite["ok"] is False
    assert lite["error_kind"] == "unknown_model"
    assert lite["status"] == 404
    assert lite["failed_stage"] == "probe"
    assert lite["hint"]
    assert "not evidence for both models" not in proc.stdout
    assert count == 3


def test_stored_profile_beats_environment(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    out = tmp_path / "stored.json"
    with StubModelServer(keys={"fx1": _STORED_FX1, "fx1-lite": _STORED_LITE}) as server:
        store = cfg / "credentials.json"
        set_endpoint("fx1", _STORED_FX1, server.url, path=store)
        set_endpoint("fx1-lite", _STORED_LITE, server.url, path=store)
        env = _child_env(
            tmp_path,
            MOONSHOT_API_KEY=_DECOY_SHARED,
            FX1_BASE_URL="http://127.0.0.1:1/v1",
            DIPCATCHER_FX1_API_KEY=_DECOY_MODEL,
            DIPCATCHER_FX1_LITE_API_KEY=_DECOY_MODEL,
        )
        proc = _run(
            tmp_path,
            ["--timeout-s", "2", "--out", str(out)],
            env=env,
        )
        seen = [(item.model, item.auth_ok, item.auth_fingerprint) for item in server.requests]
        endpoint = (server.url, server.base_url, server.port)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    document = _load_evidence(out)
    assert document["ok"] is True
    assert [row["endpoint_host"] for row in document["models"]] == ["127.0.0.1", "127.0.0.1"]
    blob = _blob(proc, out)
    _assert_secret_absent(blob, _STORED_FX1, _STORED_LITE, _DECOY_SHARED, _DECOY_MODEL)
    _assert_concrete_url_absent(blob, url=endpoint[0], base_url=endpoint[1], port=endpoint[2])
    assert "http://" not in out.read_text(encoding="utf-8")
    assert seen == [
        ("fx1", True, "...AAAA"),
        ("fx1", True, "...AAAA"),
        ("fx1-lite", True, "...BBBB"),
        ("fx1-lite", True, "...BBBB"),
    ]


def test_missing_profile_is_auth_without_network(tmp_path: Path) -> None:
    out = tmp_path / "missing.json"
    started = time.monotonic()
    proc = _run(tmp_path, ["--timeout-s", "1", "--out", str(out)], timeout=3)
    assert time.monotonic() - started < 3
    assert proc.returncode == 1, f"{proc.stdout}\n{proc.stderr}"
    document = _load_evidence(out)
    assert document["ok"] is False
    assert document["partial"] is False
    assert [row["error_kind"] for row in document["models"]] == ["auth", "auth"]
    assert all(row["failed_stage"] == "resolve" for row in document["models"])
    raw = out.read_text(encoding="utf-8")
    assert "moonshot" not in raw.lower()
    assert "http://" not in raw
    assert document["disclaimer"] in proc.stdout


def test_completion_text_is_not_recorded(tmp_path: Path) -> None:
    out = tmp_path / "text.json"

    def leak(_record: object) -> Reply:
        return Reply.ok(
            _SECRET,
            usage={"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2},
            served_model="fx1",
            headers={"x-request-id": "leak-id"},
            system_fingerprint="stub",
        )

    with StubModelServer(keys={"fx1": _SECRET}) as server:
        server.set_default(leak)
        argv = _base_argv(server, out, "--allow-partial", "--model", "fx1")
        proc = _run(tmp_path, argv, env=_child_env(tmp_path, **{_KEY_ENV: _SECRET}))
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    document = _load_evidence(out)
    assert document["models"][0]["ok"] is True
    assert document["models"][0]["identity_match"] == "match"
    _assert_secret_absent(_blob(proc, out), _SECRET)


def test_unreported_served_model_is_recorded(tmp_path: Path) -> None:
    out = tmp_path / "unreported.json"

    def omit(_record: object) -> Reply:
        return Reply.ok(
            "plain",
            usage={"total_tokens": 1},
            headers={"x-request-id": "bare-id"},
            system_fingerprint="stub",
        )

    with StubModelServer(keys={"fx1": _SECRET}) as server:
        server.set_default(omit)
        argv = _base_argv(server, out, "--allow-partial", "--model", "fx1")
        proc = _run(tmp_path, argv, env=_child_env(tmp_path, **{_KEY_ENV: _SECRET}))
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    row = _load_evidence(out)["models"][0]
    assert row["ok"] is True
    assert row["identity_match"] == "unreported"
    assert row["served_model"] is None
    assert row["request_id"] == "bare-id"
    assert row["usage"] == {"total_tokens": 1}


def test_userinfo_and_query_are_not_echoed(tmp_path: Path) -> None:
    out = tmp_path / "userinfo.json"
    hostile = f"http://user:{_SECRET}@127.0.0.1:9/v1/chat/completions?api_key={_SECRET}"
    argv = [
        "--base-url",
        hostile,
        "--api-key-env",
        _KEY_ENV,
        "--timeout-s",
        "1",
        "--out",
        str(out),
    ]
    proc = _run(tmp_path, argv, env=_child_env(tmp_path, **{_KEY_ENV: _SECRET}), timeout=3)
    assert proc.returncode == 1, f"{proc.stdout}\n{proc.stderr}"
    document = _load_evidence(out)
    assert [row["error_kind"] for row in document["models"]] == [
        "invalid_endpoint",
        "invalid_endpoint",
    ]
    assert all(row["endpoint_host"] is None for row in document["models"])
    blob = _blob(proc, out)
    _assert_secret_absent(blob, _SECRET)
    assert "127.0.0.1:9" not in blob
    assert "userinfo" in document["models"][0]["hint"]


def test_plain_http_to_public_host_is_rejected(tmp_path: Path) -> None:
    out = tmp_path / "public.json"
    argv = [
        "--base-url",
        "http://models.example.test/v1",
        "--api-key-env",
        _KEY_ENV,
        "--timeout-s",
        "1",
        "--out",
        str(out),
    ]
    started = time.monotonic()
    proc = _run(tmp_path, argv, env=_child_env(tmp_path, **{_KEY_ENV: _SECRET}), timeout=3)
    assert time.monotonic() - started < 3
    assert proc.returncode == 1, f"{proc.stdout}\n{proc.stderr}"
    document = _load_evidence(out)
    assert document["models"][0]["error_kind"] == "invalid_endpoint"
    assert "loopback" in document["models"][0]["hint"]
    _assert_secret_absent(_blob(proc, out), _SECRET)
    assert "models.example.test" not in _blob(proc, out)


@pytest.mark.parametrize(
    ("case", "kind", "status", "requests"),
    [
        ("auth", "auth", 401, 1),
        ("unknown_model", "unknown_model", 404, 1),
        ("rate_limited", "rate_limited", 429, 1),
        ("timeout", "timeout", None, 1),
        ("server", "server", 500, 1),
        ("malformed", "malformed", 200, 1),
        ("mismatch", "identity_mismatch", None, 2),
        ("redirect", "route_not_found", 302, 1),
    ],
)
def test_failure_drills(
    tmp_path: Path, case: str, kind: str, status: int | None, requests: int
) -> None:
    out = tmp_path / f"{case}.json"
    keys = {} if case == "unknown_model" else {"fx1": _SECRET}
    timeout_s = "0.4" if case == "timeout" else "2"
    proc_timeout = 6.0 if case == "timeout" else 8.0
    with StubModelServer(keys=keys) as server:
        _arm(server, case)
        argv = [
            "--base-url",
            server.url,
            "--api-key-env",
            _KEY_ENV,
            "--timeout-s",
            timeout_s,
            "--out",
            str(out),
            "--allow-partial",
            "--model",
            "fx1",
        ]
        started = time.monotonic()
        proc = _run(
            tmp_path,
            argv,
            env=_child_env(tmp_path, **{_KEY_ENV: _SECRET}),
            timeout=proc_timeout,
        )
        seen = list(server.requests)
        elapsed = time.monotonic() - started
        endpoint = (server.url, server.base_url, server.port)
    assert proc.returncode == 1, f"{case}\n{proc.stdout}\n{proc.stderr}"
    if case == "timeout":
        assert elapsed < 5
    document = _load_evidence(out)
    assert document["partial"] is True
    assert document["ok"] is False
    row = document["models"][0]
    _assert_row_shape(row)
    assert row["error_kind"] == kind
    assert row["status"] == status
    assert row["hint"]
    assert row["ok"] is False
    assert len(seen) == requests
    assert all(item.path.endswith("/v1/chat/completions") for item in seen)
    blob = _blob(proc, out)
    _assert_secret_absent(blob, _SECRET)
    _assert_concrete_url_absent(blob, url=endpoint[0], base_url=endpoint[1], port=endpoint[2])
    assert "fx1 fail host=" in proc.stdout
    assert f"fx1 {kind}:" in proc.stderr
    _assert_drill(case, row, blob, out)


def _assert_drill(case: str, row: dict[str, Any], blob: str, out: Path) -> None:
    if case == "mismatch":
        _assert_mismatch(row)
        return
    assert row["failed_stage"] == "probe"
    if case == "auth":
        assert row["request_id"] == "auth-req"
        _assert_auth_redaction(blob, out)
    elif case == "rate_limited":
        assert row["retry_after_s"] == 2
    elif case == "redirect":
        _assert_redirect(row, blob)


def _assert_auth_redaction(blob: str, out: Path) -> None:
    assert "[REDACTED]" in out.read_text(encoding="utf-8")
    assert "Authorization" not in blob


def _assert_mismatch(row: dict[str, Any]) -> None:
    assert row["probe_ok"] is True
    assert row["completion_ok"] is True
    assert row["identity_match"] == "mismatch"
    assert row["served_model"] == "other-model"
    assert row["failed_stage"] == "identity"
    assert row["request_id"] == "mismatch-2"
    assert row["probe_request_id"] == "mismatch-1"


def _assert_redirect(row: dict[str, Any], blob: str) -> None:
    assert "redirect" in row["hint"]
    assert "evil.example" not in blob


def _arm(server: StubModelServer, case: str) -> None:
    if case == "auth":
        message = (
            f"leaked {_SECRET} Authorization: Bearer {_SECRET} "
            f"http://user:{_SECRET}@evil.example/v1?api_key={_SECRET}"
        )
        server.queue(
            Reply.error(
                401,
                headers={"x-request-id": "auth-req"},
                body={"error": {"message": message, "type": "auth"}},
            )
        )
        return
    if case == "rate_limited":
        server.queue(
            Reply.error(
                429,
                headers={"Retry-After": "2"},
                body={"error": {"message": "slow down", "type": "rate_limit", "code": "rate"}},
            )
        )
        return
    if case == "timeout":
        server.queue(Reply.delay(8.0, Reply.ok("late", served_model="fx1")))
        return
    if case == "server":
        server.queue(
            Reply.error(
                500,
                body={"error": {"message": "boom", "type": "server_error", "code": "internal"}},
            )
        )
        return
    if case == "malformed":
        server.queue(Reply.malformed(_SECRET.encode("utf-8")))
        return
    if case == "mismatch":
        _MISMATCH_SEQ["n"] = 0
        server.set_default(_mismatch_reply)
        return
    if case == "redirect":
        server.queue(
            Reply.redirect(f"http://evil.example/v1/chat/completions?api_key={_SECRET}", status=302)
        )


_MISMATCH_SEQ = {"n": 0}


def _mismatch_reply(_record: object) -> Reply:
    _MISMATCH_SEQ["n"] += 1
    return Reply.ok(
        "other",
        usage={"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2},
        served_model="other-model",
        headers={"x-request-id": f"mismatch-{_MISMATCH_SEQ['n']}"},
        system_fingerprint="stub-fp",
    )


@pytest.mark.parametrize(
    ("argv", "fragment"),
    [
        (["--base-url", "http://127.0.0.1:9/v1"], "together"),
        (["--api-key-env", "SMOKE_API_KEY"], "together"),
        (["--model", "fx1"], "allow-partial"),
        (["--require-both", "--model", "fx1"], "allow-partial"),
        (["--allow-partial"], "--model"),
        (["--allow-partial", "--require-both", "--model", "fx1"], "only one"),
        (["--timeout-s", "0"], "positive"),
        (["--timeout-s", "-1"], "positive"),
        (["--timeout-s", "nan"], "positive"),
        (["--model", "nope"], "invalid choice"),
        (
            ["--api-key-env", "not a name", "--base-url", "http://127.0.0.1:9/v1"],
            "environment variable",
        ),
        (["--base-url", "   ", "--api-key-env", "SMOKE_API_KEY"], "blank"),
    ],
)
def test_usage_errors(tmp_path: Path, argv: list[str], fragment: str) -> None:
    out = tmp_path / "unused.json"
    proc = _run(tmp_path, [*argv, "--out", str(out)], timeout=3)
    assert proc.returncode == 2, f"{argv}\n{proc.stdout}\n{proc.stderr}"
    assert fragment in proc.stderr
    assert not out.exists()
    assert _SECRET not in proc.stdout + proc.stderr
