"""Offline and one-call model checks, including the release-evidence JSON record."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fx1 import __version__
from fx1.interactive.atomicstore import atomic_write_json
from fx1.interactive.model_client import ModelIdentity
from fx1.interactive.models_check import (
    CHECK_SCHEMA,
    check_models,
    format_check_report,
    format_model_list,
    list_payload,
    presence_report,
    report_exit_code,
)
from tests.fx1.stub_model_server import Reply, StubModelServer

FX1_KEY = "sk-test-fx1-secret-value-aaa111"
LITE_KEY = "sk-test-lite-secret-value-bbb222"
_CASES = ("mismatch", "auth", "missing", "limited", "timeout", "malformed")


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config = tmp_path / "cfg"
    config.mkdir()
    monkeypatch.setenv("FX1_CONFIG_DIR", str(config))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    for name in (
        "MOONSHOT_API_KEY",
        "FX1_BASE_URL",
        "MOONSHOT_API_URL",
        "DIPCATCHER_FX1_API_KEY",
        "DIPCATCHER_FX1_BASE_URL",
        "DIPCATCHER_FX1_LITE_API_KEY",
        "DIPCATCHER_FX1_LITE_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.fixture
def stub(isolated: Path) -> Iterator[StubModelServer]:
    del isolated
    server = StubModelServer({"fx1": FX1_KEY, "fx1-lite": LITE_KEY})
    server.start()
    try:
        yield server
    finally:
        server.close()


def _configure(model: str, key: str, url: str) -> None:
    from fx1.interactive.profiles import set_endpoint

    set_endpoint(model, key, url)


def _queue(server: StubModelServer, case: str, model: str) -> None:
    if case == "mismatch":
        server.queue(Reply.ok("x", served_model="other-id", usage={"total_tokens": 1}))
        return
    if case == "auth":
        body = {"error": {"message": "bad key", "type": "auth", "code": "invalid_api_key"}}
        server.queue(Reply.error(401, body=body))
        return
    if case == "missing":
        body = {
            "error": {
                "message": "model_not_found",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        }
        server.queue(Reply.error(404, body=body))
        return
    if case == "limited":
        body = {"error": {"message": "slow down", "type": "rate_limit", "code": "rate_limit"}}
        server.queue(Reply.error(429, body=body))
        return
    if case == "timeout":
        server.queue(Reply.delay(5, Reply.ok("late", served_model=model)))
        return
    server.queue(Reply.malformed())


def test_offline_configured_contacts_nothing(isolated: Path, stub: StubModelServer) -> None:
    del isolated
    _configure("fx1", FX1_KEY, stub.url)
    _configure("fx1-lite", LITE_KEY, stub.url)
    report = check_models(live=False)
    text = format_check_report(report)
    assert report_exit_code(report) == 0
    assert report.contacted is False
    assert "nothing was contacted" in text
    assert "available" not in text.casefold()
    assert stub.requests == []
    assert all(row.configured and row.url_valid and not row.contacted for row in report.models)
    blob = json.dumps(report.as_json())
    assert FX1_KEY not in blob
    assert LITE_KEY not in blob


def test_offline_missing_endpoint_exits_5(isolated: Path) -> None:
    del isolated
    report = check_models(("fx1",), live=False)
    assert report_exit_code(report) == 5
    assert report.models[0].error_kind == "no_endpoint"
    assert report.models[0].contacted is False
    assert "keys set" in (report.models[0].hint or "")


def test_list_payload_has_both_models(isolated: Path) -> None:
    del isolated
    rows, warnings = presence_report()
    payload = list_payload(rows, warnings)
    text = format_model_list(rows, warnings)
    assert payload["schema"] == "dipcatcher.models.v1"
    assert [row["model"] for row in payload["models"]] == ["fx1", "fx1-lite"]
    assert "fx1" in text and "fx1-lite" in text
    assert FX1_KEY not in text


def test_live_success_records_identity(isolated: Path, stub: StubModelServer) -> None:
    del isolated
    _configure("fx1", FX1_KEY, stub.url)
    _configure("fx1-lite", LITE_KEY, stub.url)
    moment = datetime(2026, 10, 12, 4, 0, tzinfo=UTC)
    report = check_models(live=True, timeout_s=2, clock=lambda: moment)
    assert report_exit_code(report) == 0
    assert report.fx1_version == __version__
    assert report.generated_at.startswith("2026-10-12")
    assert report.contacted is True
    payload = report.as_json()
    assert payload["schema"] == CHECK_SCHEMA
    for row in report.models:
        assert row.status == "ok"
        assert row.matched is True
        assert row.served_model == row.requested_model
        assert row.request_id
        assert row.host.startswith("127.0.0.1")
        assert row.latency_ms is not None
        assert row.usage
        assert "://" not in row.host
    blob = json.dumps(payload)
    assert FX1_KEY not in blob
    assert LITE_KEY not in blob


@pytest.mark.parametrize("model", ["fx1", "fx1-lite"])
@pytest.mark.parametrize("case", _CASES)
def test_live_failure_per_model(
    model: str, case: str, isolated: Path, stub: StubModelServer
) -> None:
    del isolated
    key = FX1_KEY if model == "fx1" else LITE_KEY
    _configure(model, key, stub.url)
    _queue(stub, case, model)
    timeout = 0.3 if case == "timeout" else 2.0
    report = check_models((model,), live=True, timeout_s=timeout)
    row = report.models[0]
    assert report_exit_code(report) == 1
    assert row.status == "failed"
    assert row.requested_model == model
    expected = {
        "mismatch": "identity_mismatch",
        "auth": "auth",
        "missing": "unknown_model",
        "limited": "rate_limited",
        "timeout": "timeout",
        "malformed": "malformed",
    }[case]
    assert row.error_kind == expected
    if case == "mismatch":
        assert row.matched is False
        assert row.served_model == "other-id"
    blob = json.dumps(report.as_json())
    assert key not in blob
    assert stub.url not in blob


def test_out_file_is_private_json(isolated: Path, stub: StubModelServer) -> None:
    del stub
    _configure("fx1", FX1_KEY, "http://127.0.0.1:9/v1/chat/completions")
    report = check_models(("fx1",), live=False)
    path = isolated / "evidence" / "models.json"
    atomic_write_json(path, report.as_json(), mode=0o600)
    assert path.stat().st_mode & 0o777 == 0o600
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema"] == CHECK_SCHEMA
    assert FX1_KEY not in path.read_text(encoding="utf-8")


def test_matched_none_does_not_fail_the_check(isolated: Path) -> None:
    del isolated
    _configure("fx1", FX1_KEY, "http://127.0.0.1:9/v1/chat/completions")

    class _Client:
        last_usage = {"total_tokens": 2}

        def ping(self, *, cancel_event: object = None) -> ModelIdentity:
            del cancel_event
            return ModelIdentity("fx1", None, "req-1", "127.0.0.1", None, None)

    def factory(**_kwargs: object) -> _Client:
        return _Client()

    report = check_models(("fx1",), live=True, timeout_s=1, client_factory=factory)
    assert report.models[0].matched is None
    assert report.models[0].status == "ok"
    assert report_exit_code(report) == 0


def test_invalid_env_url_is_not_configured(isolated: Path) -> None:
    environ = {"MOONSHOT_API_KEY": FX1_KEY, "FX1_BASE_URL": "not-a-url"}
    report = check_models(
        ("fx1",), live=False, environ=environ, path=isolated / "missing-credentials.json"
    )
    assert report.models[0].configured is False
    assert report.models[0].url_valid is False
    assert report_exit_code(report) == 5
    assert FX1_KEY not in json.dumps(report.as_json())


def test_live_failure_outranks_missing_endpoint(isolated: Path, stub: StubModelServer) -> None:
    del isolated
    _configure("fx1", "sk-wrong-key-value-not-matching", stub.url)
    report = check_models(live=True, timeout_s=2)
    kinds = {row.requested_model: row.error_kind for row in report.models}
    assert kinds["fx1"] == "auth"
    assert kinds["fx1-lite"] == "no_endpoint"
    assert report_exit_code(report) == 1
