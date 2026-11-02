"""FX1_LOCAL_SERVE_URL is a loopback http(s) URL unless explicitly widened."""

from __future__ import annotations

import urllib.parse

import pytest

from fx1.serve.backends import (
    LOCAL_SERVE_ALLOW_NON_LOOPBACK_ENV,
    LOCAL_SERVE_URL_ENV,
    _require_local_serve_destination,
)


def _parse(url: str) -> urllib.parse.ParseResult:
    return urllib.parse.urlparse(url)


def _require(url: str) -> None:
    parsed = _parse(url)
    assert parsed.scheme in ("http", "https") and parsed.netloc
    _require_local_serve_destination(parsed)


def test_loopback_urls_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LOCAL_SERVE_ALLOW_NON_LOOPBACK_ENV, raising=False)
    for url in (
        "http://127.0.0.1:9/v1",
        "https://127.0.0.1/v1",
        "http://[::1]/v1",
        "http://[::ffff:127.0.0.1]:8080/v1",
        "http://localhost/v1",
        "http://localhost./v1",
    ):
        _require(url)


def test_userinfo_port_and_non_loopback_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LOCAL_SERVE_ALLOW_NON_LOOPBACK_ENV, raising=False)
    with pytest.raises(RuntimeError, match="must not embed userinfo") as raised:
        _require("http://user:secret@127.0.0.1/v1")
    assert "secret" not in str(raised.value)
    assert "user:secret" not in str(raised.value)
    for url in ("http://127.0.0.1:0/v1", "http://127.0.0.1:99999/v1"):
        with pytest.raises(RuntimeError, match="invalid port"):
            _require(url)
    for url in ("http://10.0.0.1/v1", "http://example.com/v1", "https://8.8.8.8/v1"):
        with pytest.raises(RuntimeError, match=LOCAL_SERVE_ALLOW_NON_LOOPBACK_ENV):
            _require(url)


def test_non_loopback_opt_in_still_blocks_special_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOCAL_SERVE_ALLOW_NON_LOOPBACK_ENV, "yes")
    _require("https://example.com:8443/v1")
    _require("http://10.1.2.3/v1")
    for url in (
        "http://169.254.169.254/latest/meta-data/",
        "http://[fe80::1]/v1",
        "http://224.0.0.1/v1",
        "http://192.0.2.1/v1",
        "http://192.88.99.1/v1",
        "http://100.64.0.1/v1",
        "http://0.0.0.0/v1",
        "http://[ff02::1]/v1",
    ):
        with pytest.raises(RuntimeError, match="destination is not allowed"):
            _require(url)
    assert LOCAL_SERVE_URL_ENV == "FX1_LOCAL_SERVE_URL"
