"""Credential redaction for logs, reports, and exception text.

SYNTHETIC secrets only. Assertions require the secret substring to be absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.data.concurrent_io import IoError, pooled_request
from quant_fund.data.sources.base import HttpClient, SourceError
from quant_fund.utils.logging import _redact_processor
from quant_fund.utils.secrets import (
    REDACTED,
    is_secret_field,
    redact_log_value,
    redact_obj,
    redact_text,
)

_USER = "user:s3cret-synthetic"
_QUERY = "SECRETKEY"
_BEARER = "sk-syntheticabcdefgh"
_WEBHOOK = f"https://{_USER}@hooks.example/hook?token={_QUERY}"
_DSN = "https://syntheticpublic@o.ingest.sentry.io/1"
_POSTGRES = "postgres://app:s3cret-synthetic@db.internal/lab"


def test_field_names_drop_credentials_and_keep_identifiers() -> None:
    assert is_secret_field("api_key")
    assert is_secret_field("api-key")
    assert is_secret_field("password")
    assert is_secret_field("password_hash")
    assert is_secret_field("dsn")
    assert is_secret_field("callback_url")
    assert is_secret_field("database_url")
    assert is_secret_field("access_token")
    assert not is_secret_field("key_id")
    assert not is_secret_field("max_tokens")
    assert not is_secret_field("tokens_used")
    assert redact_log_value("api_key", _BEARER) == REDACTED
    assert redact_log_value("key_id", "visible") == "visible"
    assert redact_log_value("max_tokens", 128) == 128


def test_redact_text_scrubs_userinfo_query_bearer_and_prefixes() -> None:
    scrubbed = redact_text(
        f"dsn {_DSN} db {_POSTGRES} hook {_WEBHOOK} Authorization: Bearer {_BEARER} "
        f"whsec-synthetic1 fx1k_synthetic1 https://example.test/public"
    )
    assert "s3cret-synthetic" not in scrubbed
    assert "syntheticpublic" not in scrubbed
    assert _QUERY not in scrubbed
    assert _BEARER not in scrubbed
    assert "whsec-synthetic1" not in scrubbed
    assert "fx1k_synthetic1" not in scrubbed
    assert "https://example.test/public" in scrubbed
    assert f"https://{REDACTED}@" in scrubbed
    assert "token=[redacted]" in scrubbed
    assert f"Bearer {REDACTED}" in scrubbed
    assert redact_text("the token count is 12") == "the token count is 12"


def test_redact_obj_walks_exceptions_and_nested_values() -> None:
    exc = ValueError(f"callback {_WEBHOOK}")
    scrubbed = redact_obj(
        {
            "error": exc,
            "nested": [_DSN, {"callback_url": _WEBHOOK, "key_id": "visible"}],
        }
    )
    blob = str(scrubbed)
    assert "s3cret-synthetic" not in blob
    assert "syntheticpublic" not in blob
    assert _QUERY not in blob
    assert "ValueError" in blob
    assert scrubbed["nested"][1]["callback_url"] == REDACTED
    assert scrubbed["nested"][1]["key_id"] == "visible"


def test_structlog_processor_scrubs_event_dict() -> None:
    event = {
        "event": f"fetch {_WEBHOOK}",
        "url": _WEBHOOK,
        "exception": f"URLError: {_DSN}",
        "api_key": _BEARER,
        "key_id": "visible",
    }
    out = _redact_processor(None, "warning", event)
    blob = str(out)
    assert "s3cret-synthetic" not in blob
    assert "syntheticpublic" not in blob
    assert _QUERY not in blob
    assert _BEARER not in blob
    redacted_key = out["api_key"]
    assert redacted_key == REDACTED
    assert out["key_id"] == "visible"


def test_http_errors_drop_query_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SourceError) as unsupported:
        HttpClient()._request(f"ftp://apps.example.test/api?UserID={_QUERY}&method=GETDATA")
    text = str(unsupported.value)
    assert _QUERY not in text
    assert "UserID=[redacted]" in text

    def boom(*_args: object, **_kwargs: object) -> bytes:
        raise IoError(f"GET {_WEBHOOK} failed")

    monkeypatch.setattr("quant_fund.data.sources.base.pooled_request", boom)
    with pytest.raises(SourceError) as transport:
        HttpClient(retries=0)._request("https://api.example.test/v1")
    assert _QUERY not in str(transport.value)
    assert "s3cret-synthetic" not in str(transport.value)


def test_pooled_request_rejects_credential_url_before_network() -> None:
    with pytest.raises(IoError) as exc:
        pooled_request(
            "GET",
            f"ftp://{_USER}@host.example/x?token={_QUERY}",
            timeout=1.0,
            max_bytes=100,
        )
    text = str(exc.value)
    assert "s3cret-synthetic" not in text
    assert _QUERY not in text
    assert "unsupported URL" in text


def test_harness_and_batch_errors_do_not_echo_urls() -> None:
    from fx1.serve.client import HarnessClient
    from fx1.serve.openai_compat import OpenAICompatError, batch_line_shape

    with pytest.raises(ValueError) as harness:
        HarnessClient(f"ftp://{_USER}@host.example/v1?api_key={_QUERY}")
    assert "s3cret-synthetic" not in str(harness.value)
    assert _QUERY not in str(harness.value)

    line = {
        "custom_id": "c1",
        "method": "POST",
        "url": _WEBHOOK,
        "body": {},
    }
    with pytest.raises(OpenAICompatError) as batch:
        batch_line_shape(
            line,
            endpoint=f"https://{_USER}@other.example/v1?token={_QUERY}",
            lineno=1,
        )
    assert "s3cret-synthetic" not in str(batch.value)
    assert _QUERY not in str(batch.value)


def test_vendor_base_url_error_drops_userinfo() -> None:
    from quant_fund.data.adapters.vendor_http import (
        DEFAULT_API_KEY_ENV,
        VendorEntitlementError,
        VendorHttpAdapter,
        VendorHttpConfig,
    )

    config = VendorHttpConfig(
        vendor="example_vendor",
        base_url=f"ftp://{_USER}@vendor.example/v1?api_key={_QUERY}",
        license_acknowledged=True,
        environ={DEFAULT_API_KEY_ENV: "synthetic-key"},
    )
    with pytest.raises(VendorEntitlementError) as exc:
        VendorHttpAdapter(config, transport=lambda *_a, **_k: b"{}")
    text = str(exc.value)
    assert "s3cret-synthetic" not in text
    assert _QUERY not in text
    assert "HTTP(S)" in text


def test_dataset_and_timestamp_errors_drop_credentials(tmp_path: Path) -> None:
    from quant_fund.data.adapters.hf_ohlcv_1m import stream_https
    from quant_fund.research.timestamp_anchor import stamp_timestamp

    url = f"http://{_USER}@example.invalid/data?api_key={_QUERY}"
    with pytest.raises(SourceError) as dataset:
        stream_https(url, tmp_path / "f", opener=lambda *_a, **_k: None, max_bytes=10, timeout=1)
    assert "s3cret-synthetic" not in str(dataset.value)
    assert _QUERY not in str(dataset.value)
    assert "non-HTTPS" in str(dataset.value)

    target = tmp_path / "pin.txt"
    target.write_text("x")
    with pytest.raises(ValueError) as stamped:
        stamp_timestamp(
            target.name,
            root=tmp_path,
            tsr_url=f"ftp://{_USER}@tsa.example/stamp?token={_QUERY}",
        )
    assert "s3cret-synthetic" not in str(stamped.value)
    assert _QUERY not in str(stamped.value)
    assert "http(s)" in str(stamped.value)


def test_refresh_report_redacts_source_credentials(tmp_path: Path) -> None:
    from fx1.flash.refresh import refresh
    from fx1.flash.store import FlashStore

    store = FlashStore(tmp_path / "entries.jsonl")
    entry = store.add(
        text="synthetic finding",
        sources=[{"url": _WEBHOOK}],
    )

    def fetch(_url: str) -> object:
        raise RuntimeError(f"Bearer {_BEARER}")

    report = refresh(store, entry.id, fetch_fn=fetch)
    assert report is not None
    blob = (report.note or "") + " ".join(report.checked) + " ".join(report.unreachable)
    assert "s3cret-synthetic" not in blob
    assert _QUERY not in blob
    assert _BEARER not in blob
    assert REDACTED in blob
