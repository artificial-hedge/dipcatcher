"""Fuzz auth and rate-limit middleware: bad headers, huge keys, unicode, request ids."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from hypothesis import example, given, settings
from hypothesis import strategies as st

from quant_fund.api.app import app as lab_app
from quant_fund.api.http_controls import key_rate_response, reset_rate_gates, scoped_auth_decision
from quant_fund.api.research_api import ResearchApiSettings, create_app
from tests.unit.api.test_http_controls_fuzz import _assert_clean, _invoke

_ADMIN = "fuzz-admin-key"
_ENV = (
    "QUANT_API_KEY",
    "QUANT_API_READ_KEY",
    "QUANT_API_AUDIT_LOG",
    "QUANT_API_AUDIT_READS",
    "QUANT_API_RATE_PER_SEC",
    "QUANT_API_RATE_BURST",
    "QUANT_API_LOOPBACK_RATE_PER_SEC",
    "QUANT_API_LOOPBACK_RATE_BURST",
    "RESEARCH_API_KEY",
    "RESEARCH_API_READ_KEY",
    "RESEARCH_API_AUDIT_LOG",
    "RESEARCH_API_RATE_PER_SEC",
    "RESEARCH_API_RATE_BURST",
    "RESEARCH_API_LOOPBACK_RATE_PER_SEC",
    "RESEARCH_API_LOOPBACK_RATE_BURST",
    "DIPCATCHER_RATE_BACKEND",
    "DIPCATCHER_RATE_STORE",
)
_RESEARCH: Any = None


def _research() -> Any:
    global _RESEARCH
    if _RESEARCH is None:
        root = Path(tempfile.mkdtemp(prefix="auth-rate-fuzz-"))
        for name in ("data", "receipts", "verifier", "artifacts"):
            (root / name).mkdir()
        _RESEARCH = create_app(
            ResearchApiSettings(
                data_root=root / "data",
                receipts_dir=root / "receipts",
                verifier_dir=root / "verifier",
                artifacts_dir=root / "artifacts",
                api_key=_ADMIN,
            )
        )
    return _RESEARCH


def _saved() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in _ENV}


def _restore(saved: dict[str, str | None]) -> None:
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _open_limits() -> None:
    os.environ["QUANT_API_KEY"] = _ADMIN
    os.environ.pop("QUANT_API_READ_KEY", None)
    os.environ.pop("QUANT_API_AUDIT_LOG", None)
    os.environ.pop("RESEARCH_API_AUDIT_LOG", None)
    os.environ.pop("DIPCATCHER_RATE_BACKEND", None)
    os.environ.pop("DIPCATCHER_RATE_STORE", None)
    for name in (
        "QUANT_API_RATE_PER_SEC",
        "QUANT_API_RATE_BURST",
        "QUANT_API_LOOPBACK_RATE_PER_SEC",
        "QUANT_API_LOOPBACK_RATE_BURST",
        "RESEARCH_API_RATE_PER_SEC",
        "RESEARCH_API_RATE_BURST",
        "RESEARCH_API_LOOPBACK_RATE_PER_SEC",
        "RESEARCH_API_LOOPBACK_RATE_BURST",
    ):
        os.environ[name] = "1000"
    reset_rate_gates()


def _peer(host: str) -> tuple[str, int]:
    return (host, 9)


class _Client:
    def __init__(self, host: str) -> None:
        self.host = host


class _Request:
    def __init__(self, host: str) -> None:
        self.client = _Client(host)


@settings(max_examples=20, deadline=None)
@given(
    key=st.one_of(
        st.binary(min_size=0, max_size=4096),
        st.text(max_size=48).map(lambda text: text.encode("utf-8", "replace")),
    ),
    bad=st.one_of(st.none(), st.sampled_from([b"\r", b"\n", b"\x00", b"\r\nX-Injected: pwned"])),
    request_ids=st.lists(
        st.one_of(
            st.binary(min_size=0, max_size=160),
            st.text(max_size=32).map(lambda text: text.encode("utf-8", "replace")),
        ),
        min_size=2,
        max_size=4,
    ),
    host=st.text(
        alphabet=st.characters(min_codepoint=33, max_codepoint=1024),
        min_size=1,
        max_size=40,
    ),
)
@example(
    key="clé-密钥".encode(),
    bad=b"\r\n",
    request_ids=[b"req-a", b"req-a"],
    host="198.51.100.20",
)
@example(key=b"k" * 2048, bad=b"\x00", request_ids=[b"one", b"two", b"one"], host="localhost")
def test_malformed_headers_huge_keys_and_repeated_request_ids(
    key: bytes,
    bad: bytes | None,
    request_ids: list[bytes],
    host: str,
) -> None:
    saved = _saved()
    try:
        _open_limits()
        headers = [(b"host", b"example.com"), (b"x-api-key", key)]
        if bad is not None:
            headers.append((b"x-marker", bad))
        for item in request_ids:
            headers.append((b"x-request-id", item))
        for target, path in ((lab_app, "/models"), (_research(), "/receipts")):
            status, body, response_headers = _invoke(
                target,
                method="GET",
                path=path,
                headers=headers,
                chunks=[b""],
                client=_peer(host),
            )
            _assert_clean(status, body, response_headers)
            assert status < 500
            assert _ADMIN.encode() not in body
            for name, value in response_headers:
                assert _ADMIN.encode() not in value
                assert b"\r" not in name and b"\n" not in name
            if len(key) > 8:
                assert key not in body
                for _name, value in response_headers:
                    assert key not in value
    finally:
        _restore(saved)
        reset_rate_gates()


def test_a_huge_unicode_key_is_rejected_without_an_echo() -> None:
    saved = _saved()
    try:
        _open_limits()
        huge = b"z" * 100_000
        unicode_key = "clé-密钥-\u2603".encode()
        for key in (huge, unicode_key):
            headers = [
                (b"host", b"example.com"),
                (b"x-api-key", key),
                (b"x-request-id", key[:120]),
                (b"x-request-id", b"second-id"),
            ]
            status, body, response_headers = _invoke(
                lab_app,
                method="GET",
                path="/models",
                headers=headers,
                chunks=[b""],
            )
            assert status == 401
            assert json.loads(body) == {"detail": "invalid or missing X-API-Key"}
            assert key not in body
            _assert_clean(status, body, response_headers)
    finally:
        _restore(saved)


def test_repeated_request_ids_stay_out_of_the_audit_line(tmp_path: Path) -> None:
    saved = _saved()
    log_path = tmp_path / "audit.jsonl"
    try:
        _open_limits()
        os.environ["QUANT_API_AUDIT_LOG"] = str(log_path)
        headers = [
            (b"host", b"example.com"),
            (b"x-api-key", _ADMIN.encode()),
            (b"x-request-id", _ADMIN.encode()),
            (b"x-request-id", b"second-id"),
        ]
        status, body, response_headers = _invoke(
            lab_app,
            method="GET",
            path="/backtest/" + "ab" * 8,
            headers=headers,
            chunks=[b""],
        )
        assert status == 404
        assert _ADMIN.encode() not in body
        _assert_clean(status, body, response_headers)
        text = log_path.read_text(encoding="utf-8")
        assert _ADMIN not in text
        row = json.loads(text)
        assert row["outcome"] == "read"
        assert row["kind"] == "ledger"
        assert row["request_id"] != _ADMIN
    finally:
        _restore(saved)


@settings(max_examples=20, deadline=None)
@given(
    text=st.text(
        alphabet=st.characters(min_codepoint=0xD800, max_codepoint=0xDFFF),
        min_size=1,
        max_size=8,
    )
)
def test_surrogate_keys_do_not_raise(text: str) -> None:
    saved = _saved()
    try:
        _open_limits()
        assert scoped_auth_decision("GET", text, _ADMIN, None) == "unauthorized"
        assert scoped_auth_decision("POST", text, _ADMIN, "read-key") == "unauthorized"
        limited = key_rate_response(_Request("203.0.113.9"), "QUANT_API", text)
        assert limited is None or limited.status_code in {429, 503}
    finally:
        _restore(saved)
