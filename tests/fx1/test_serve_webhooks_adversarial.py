"""SYNTHETIC adversarial probes for ``fx1.serve.webhooks``.

Signed-delivery correctness (HMAC over ``<ts>.<raw body>``), bounded
fire-once semantics (4xx terminal, 5xx/redirect retried but never
followed), private-network scoping (opt-in env only inside set+restore
guards), and receiver-side verification — all on a loopback HTTP sink
with synthetic secrets. No external network is touched.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from fx1.serve.webhooks import (
    WEBHOOK_SIGNATURE_HEADER,
    WEBHOOK_TIMESTAMP_HEADER,
    check_callback_url,
    deliver_signed,
    sign_webhook,
    verify_webhook,
)

_SECRET = "synthetic-webhook-secret"
_BODY = b'{"job_id":"synthetic","status":"finished"}'


class _Hit:
    def __init__(self, headers: Any, body: bytes) -> None:
        self.headers = headers
        self.body = body


class _Sink:
    """Loopback POST recorder answering a fixed status (or redirect)."""

    def __init__(self, status: int = 200, location: str | None = None) -> None:
        self.status = status
        self.location = location
        self.hits: list[_Hit] = []

        sink = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — stdlib hook name
                length = int(self.headers.get("Content-Length") or 0)
                sink.hits.append(_Hit(self.headers, self.rfile.read(length)))
                self.send_response(sink.status)
                if sink.location is not None:
                    self.send_header("Location", sink.location)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *_args: Any) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/hook?x=1"

    def close(self) -> None:
        self.server.shutdown()
        self.thread.join(5)
        self.server.server_close()

    def __enter__(self) -> _Sink:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


@pytest.fixture
def private_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope the private-network opt-in to this test's own delivery."""
    monkeypatch.setenv("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", "1")


def _verify_hit(secret: str, hit: _Hit) -> bool:
    return verify_webhook(
        secret,
        hit.headers.get(WEBHOOK_TIMESTAMP_HEADER),
        hit.headers.get(WEBHOOK_SIGNATURE_HEADER),
        hit.body,
        tolerance_s=300,
    )


def test_signed_delivery_roundtrip_receiver_verifies(
    private_opt_in: None,
) -> None:
    with _Sink() as sink:
        delivered, error, attempts = deliver_signed(
            sink.url, _SECRET, _BODY, backoff_s=0, timeout_s=5
        )
        assert delivered is True and error is None and attempts == 1
    assert len(sink.hits) == 1
    hit = sink.hits[0]
    assert hit.body == _BODY
    assert hit.headers.get("Content-Type") == "application/json"
    ts = hit.headers.get(WEBHOOK_TIMESTAMP_HEADER)
    sig = hit.headers.get(WEBHOOK_SIGNATURE_HEADER)
    assert ts is not None and int(ts) > 0
    assert sig is not None and sig.startswith("sha256=")
    # Independent receiver-side recompute — not the module's own sign path.
    expected = hmac.new(_SECRET.encode(), f"{ts}.".encode() + _BODY, hashlib.sha256).hexdigest()
    assert sig == f"sha256={expected}"
    assert _verify_hit(_SECRET, hit) is True
    assert _verify_hit("synthetic-other-secret", hit) is False
    tampered = _Hit(hit.headers, _BODY + b" ")
    assert _verify_hit(_SECRET, tampered) is False


def test_unsigned_delivery_carries_no_signature_headers(
    private_opt_in: None,
) -> None:
    """No secret → an unsigned POST; receivers cannot authenticate it."""
    with _Sink() as sink:
        delivered, _error, attempts = deliver_signed(
            sink.url, None, _BODY, backoff_s=0, timeout_s=5
        )
    assert delivered is True and attempts == 1
    hit = sink.hits[0]
    assert hit.headers.get(WEBHOOK_SIGNATURE_HEADER) is None
    assert hit.headers.get(WEBHOOK_TIMESTAMP_HEADER) is None
    assert _verify_hit(_SECRET, hit) is False


def test_client_rejection_is_terminal_not_retried(private_opt_in: None) -> None:
    with _Sink(status=400) as sink:
        delivered, error, attempts = deliver_signed(
            sink.url, _SECRET, _BODY, max_attempts=3, backoff_s=0, timeout_s=5
        )
    assert delivered is False and attempts == 1
    assert error == "callback endpoint returned 400"
    assert len(sink.hits) == 1


def test_server_faults_retry_but_stay_bounded(private_opt_in: None) -> None:
    with _Sink(status=503) as sink:
        delivered, error, attempts = deliver_signed(
            sink.url, _SECRET, _BODY, max_attempts=2, backoff_s=0, timeout_s=5
        )
    assert delivered is False and attempts == 2
    assert error == "callback endpoint returned 503"
    assert len(sink.hits) == 2


def test_redirect_is_never_followed(private_opt_in: None) -> None:
    """A 3xx is not followed — the redirect target sees zero requests."""
    with _Sink() as target, _Sink(status=302, location=target.url) as redirector:
        delivered, error, _attempts = deliver_signed(
            redirector.url, _SECRET, _BODY, max_attempts=1, backoff_s=0, timeout_s=5
        )
    assert delivered is False
    assert error == "callback endpoint returned 302"
    assert len(redirector.hits) == 1
    assert target.hits == []


def test_signature_binds_raw_bytes_not_reserialized_content() -> None:
    """The signature covers the exact wire body — a JSON-equal body with
    different whitespace does not authenticate."""
    sig = sign_webhook(_SECRET, "1000", _BODY)
    assert verify_webhook(_SECRET, "1000", sig, _BODY, now=1000, tolerance_s=300) is True
    reserialized = json.dumps(json.loads(_BODY), indent=2).encode()
    assert reserialized != _BODY
    assert verify_webhook(_SECRET, "1000", sig, reserialized, now=1000, tolerance_s=300) is False


def test_verification_is_stateless_inside_the_freshness_window() -> None:
    """verify_webhook holds no replay table: the same delivery verifies twice
    inside tolerance (receivers dedupe at a higher layer) and dies outside it."""
    sig = sign_webhook(_SECRET, "1000", _BODY)
    for _ in range(2):
        assert verify_webhook(_SECRET, "1000", sig, _BODY, now=1299, tolerance_s=300) is True
    assert verify_webhook(_SECRET, "1000", sig, _BODY, now=1301, tolerance_s=300) is False


def test_wrong_scheme_userinfo_and_hostless_urls_fail_closed() -> None:
    for url in (
        "file:///etc/passwd",
        "gopher://example.com/x",
        "ftp://example.com/x",
        "http://user:pass@example.com/hook",
        "http://user@example.com/hook",
        "http:///no-host",
        "not a url",
        "",
    ):
        with pytest.raises(ValueError):
            check_callback_url(url)
    assert check_callback_url(None) is None
    assert check_callback_url("https://example.com/hook") == "https://example.com/hook"


def test_deliver_signed_never_raises_on_malformed_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", raising=False)
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic"))
    )
    for url in (":::garbage", "http://[::1", "https://unresolvable.invalid/hook"):
        delivered, error, attempts = deliver_signed(url, _SECRET, _BODY, backoff_s=0)
        assert delivered is False and isinstance(attempts, int)
        assert error is not None


def test_legacy_special_use_destinations_refused_even_with_opt_in(
    private_opt_in: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """192.88.99.0/24 (6a44 relay) and fec0::/10 (IPv6 site-local) classify
    as global on this Python — deprecated, unroutable space the private
    opt-in does NOT rescue (same destination policy as BYOK backends)."""
    for url in ("http://192.88.99.2/hook", "http://[fec0::1]/hook"):
        with pytest.raises(ValueError, match="deprecated special-use"):
            check_callback_url(url)
    # A public-looking hostname resolving to legacy space dies at delivery.
    monkeypatch.setattr(
        "fx1.serve.webhooks.socket.getaddrinfo",
        lambda *_a, **_k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.88.99.2", 80))],
    )
    delivered, error, _attempts = deliver_signed(
        "https://cb.example/hook", _SECRET, _BODY, max_attempts=1, backoff_s=0
    )
    assert delivered is False and "deprecated special-use" in (error or "")


def test_private_network_opt_in_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside the opt-in guard the private literal refuses again."""
    monkeypatch.delenv("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", raising=False)
    with pytest.raises(ValueError, match="private or special-use"):
        check_callback_url("http://127.0.0.1/hook")
    with monkeypatch.context() as scoped:
        scoped.setenv("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", "1")
        assert check_callback_url("http://127.0.0.1/hook") == "http://127.0.0.1/hook"
    with pytest.raises(ValueError, match="private or special-use"):
        check_callback_url("http://127.0.0.1/hook")


def test_mixed_address_health_fails_over_within_one_attempt(
    private_opt_in: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead validated address falls through to the next without burning a
    retry attempt."""
    calls: list[str] = []

    def post(_parsed: Any, address: str, *_args: Any, **_kwargs: Any) -> int:
        calls.append(address)
        if address == "10.1.1.1":
            raise OSError("synthetic connect failure")
        return 204

    monkeypatch.setattr(
        "fx1.serve.webhooks._resolved_addresses",
        lambda _host, _port: ("10.1.1.1", "10.1.1.2"),
    )
    monkeypatch.setattr("fx1.serve.webhooks._post_once", post)
    delivered, error, attempts = deliver_signed(
        "http://127.0.0.1/hook", _SECRET, _BODY, max_attempts=1, backoff_s=0
    )
    assert delivered is True and error is None and attempts == 1
    assert calls == ["10.1.1.1", "10.1.1.2"]
