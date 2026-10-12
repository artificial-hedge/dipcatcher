"""Endpoint URL checks, redirects, proxy userinfo, and Retry-After bounds."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from fx1.interactive.endpoint import EndpointError, normalize_endpoint
from fx1.interactive.model_client import ModelClient, RetryPolicy, choose_proxy
from fx1.interactive.profiles import set_endpoint
from tests.fx1.redteam_support import SECRET, isolate
from tests.fx1.stub_model_server import Reply, StubModelServer


def test_hostile_endpoint_text_is_rejected_without_echoing_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate(monkeypatch, tmp_path)
    samples = (
        f"http://user:{SECRET}@127.0.0.1:9/v1",
        "http://example.com/v1",
        "http://127.0.0.1%00.example/v1",
        "http://xn--e1awd7f.example/v1",
        "http://ex\u0430mple.com/v1",
        "http://[::1%eth0]/v1",
        "http://127.0.0.1:9/v1\r\nX-Injected: 1",
    )
    for raw in samples:
        with pytest.raises((EndpointError, ValueError)) as caught:
            set_endpoint("fx1", SECRET, raw)
        assert SECRET not in str(caught.value)
        assert "X-Injected" not in str(caught.value)
    with pytest.raises(ValueError) as caught:
        set_endpoint("fx1", f"{SECRET}\r\nX-Injected: 1", "https://models.example.test/v1")
    assert SECRET not in str(caught.value)
    spec = normalize_endpoint("http://127.0.0.1:9/v1")
    assert spec.host == "127.0.0.1"
    assert normalize_endpoint("http://[::1]/v1").origin == "http://[::1]"


def test_redirect_does_not_follow_or_send_the_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate(monkeypatch, tmp_path)
    bait_hits: list[str] = []

    class Bait(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or "0")
            self.rfile.read(max(0, length))
            bait_hits.append(self.path)
            self.send_response(204)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            return

    bait = ThreadingHTTPServer(("127.0.0.1", 0), Bait)
    bait_port = bait.server_address[1]

    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or "0")
            self.rfile.read(max(0, length))
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{bait_port}/stolen")
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            return

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threads = [
        threading.Thread(target=bait.serve_forever, daemon=True),
        threading.Thread(target=origin.serve_forever, daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        port = origin.server_address[1]
        client = ModelClient(
            api_key=SECRET,
            model="fx1",
            api_url=f"http://127.0.0.1:{port}/v1/chat/completions",
            timeout_s=2.0,
            retry=RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0, jitter=0.0),
        )
        with pytest.raises(Exception) as caught:
            client.complete([{"role": "user", "content": "ping"}])
    finally:
        origin.shutdown()
        bait.shutdown()
        origin.server_close()
        bait.server_close()
    assert SECRET not in str(caught.value)
    assert bait_hits == []


def test_retry_after_huge_or_negative_does_not_sleep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate(monkeypatch, tmp_path)
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        if seconds > 5:
            raise AssertionError(f"retry slept {seconds}")
        slept.append(seconds)

    server = StubModelServer({"fx1": SECRET})
    server.start()
    try:
        server.queue(Reply.error(429, headers={"Retry-After": "1000000000"}))
        server.queue(Reply.error(429, headers={"Retry-After": "-5"}))
        client = ModelClient(
            api_key=SECRET,
            model="fx1",
            api_url=server.url,
            timeout_s=2.0,
            retry=RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0, jitter=0.0),
            sleep=sleep,
        )
        with pytest.raises(Exception) as first:
            client.complete([{"role": "user", "content": "ping"}])
        with pytest.raises(Exception) as second:
            client.complete([{"role": "user", "content": "ping"}])
    finally:
        server.close()
    assert SECRET not in str(first.value)
    assert SECRET not in str(second.value)
    assert slept == []


def test_proxy_userinfo_is_not_a_connection_target() -> None:
    proxy = f"http://user:{SECRET}@127.0.0.1:9"
    chosen = choose_proxy(
        "models.example.test",
        "https",
        proxies={"https": proxy},
        bypass=lambda _host: False,
    )
    assert chosen == ("127.0.0.1", 9)
    assert choose_proxy("127.0.0.1", "http", proxies={"http": proxy}) is None


def test_stub_request_body_does_not_carry_the_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate(monkeypatch, tmp_path)
    server = StubModelServer({"fx1": SECRET})
    server.start()
    try:
        server.queue(Reply.ok("SYNTHETIC qlike 0.2", served_model="fx1"))
        client = ModelClient(
            api_key=SECRET,
            model="fx1\r\nX-Injected: 1",
            api_url=server.url,
            timeout_s=2.0,
            retry=RetryPolicy(max_attempts=1),
        )
        text = client.complete([{"role": "user", "content": "ping"}])
        recorded = server.requests
    finally:
        server.close()
    assert "qlike" in text
    assert recorded
    raw = json.dumps(recorded[0].body)
    assert SECRET not in raw
    assert "\r\nX-Injected" not in raw
