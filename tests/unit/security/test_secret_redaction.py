"""Credential material stays out of logs, reprs, and crash traces."""

from __future__ import annotations

import logging
import sys

import pytest

from fx1.secret_guard import install_secret_redaction, redact_text, redacting_excepthook
from fx1.serve.backends import HostedK3Backend
from fx1.serve.client import HarnessClient
from quant_fund.live.brokers import AlpacaBroker
from quant_fund.observe.logging import log_event, set_log_sink

_MARKER = "sk-" + "unitredactsecret"
_ENV_MARKER = "moonshot-env-secret-value"
_HOST = "https://example.test/v1/chat/completions"


def test_assigned_secret_is_redacted() -> None:
    assert _MARKER not in redact_text(f"api_key={_MARKER}")
    assert _MARKER not in redact_text(f"Authorization: Bearer {_MARKER}")
    assert "[redacted]" in redact_text(f"Authorization: Bearer {_MARKER}")
    assert redact_text(f"Bearer {_MARKER}") == "Bearer [redacted]"


def test_child_and_root_loggers_redact(caplog: pytest.LogCaptureFixture) -> None:
    install_secret_redaction()
    caplog.set_level(logging.INFO)
    logging.getLogger("fx1.child").error("api_key=%s", _MARKER)
    logging.getLogger().error("authorization=%s", _MARKER)
    assert _MARKER not in caplog.text
    assert "[redacted]" in caplog.text


def test_env_secret_is_redacted_without_a_label(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", _ENV_MARKER)
    install_secret_redaction()
    caplog.set_level(logging.ERROR)
    logging.getLogger("fx1.child").error("upstream returned %s", _ENV_MARKER)
    assert _ENV_MARKER not in caplog.text
    assert "[redacted]" in caplog.text


def test_logged_exception_redacts_the_message(caplog: pytest.LogCaptureFixture) -> None:
    install_secret_redaction()
    caplog.set_level(logging.ERROR)
    try:
        raise RuntimeError(f"api_key={_MARKER}")
    except RuntimeError:
        logging.getLogger("fx1.child").exception("request failed")
    assert _MARKER not in caplog.text


def test_excepthook_redacts(capsys: pytest.CaptureFixture[str]) -> None:
    try:
        raise RuntimeError(f"api_key={_MARKER}")
    except RuntimeError:
        exc_type, exc, tb = sys.exc_info()
        assert exc_type is not None and exc is not None
        redacting_excepthook(exc_type, exc, tb)
    captured = capsys.readouterr()
    assert _MARKER not in captured.err
    assert "[redacted]" in captured.err


def test_repr_reports_presence_only() -> None:
    backend = HostedK3Backend(api_key=_MARKER, api_url=_HOST)
    client = HarnessClient("http://127.0.0.1:9", api_key=_MARKER)
    broker = AlpacaBroker("alpaca-paper", (_MARKER, _ENV_MARKER))
    rendered = " ".join((repr(backend), repr(client), repr(broker)))
    assert _MARKER not in rendered
    assert _ENV_MARKER not in rendered
    assert "credential=set" in rendered


def test_configuration_error_does_not_echo_another_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", _ENV_MARKER)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as raised:
        HostedK3Backend()
    assert _ENV_MARKER not in str(raised.value)


def test_log_event_redacts_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    sink: list[dict[str, object]] = []
    set_log_sink(sink)
    try:
        log_event("stage", note=f"saw {_MARKER}", key_id="visible")
    finally:
        set_log_sink(None)
    assert sink
    assert sink[0]["key_id"] == "visible"
    assert _MARKER not in str(sink[0]["note"])
