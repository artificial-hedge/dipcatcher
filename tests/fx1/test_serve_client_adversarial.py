"""SYNTHETIC adversarial probes for fx1.serve.client — transport seams,
credential hygiene, URL validation, and wire-shape honesty."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from fx1.serve.client import HarnessClient, HarnessTransportError

_MESSAGES = [{"role": "user", "content": "SYNTHETIC probe"}]


class _QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        pass


def _server(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def test_credentialed_get_never_follows_a_redirect() -> None:
    """A 30x on a credentialed call must fail closed: urllib would replay
    the caller-supplied ``X-API-Key`` to the redirect target by default."""
    reached: list[str] = []

    class Target(_QuietHandler):
        def do_GET(self) -> None:  # noqa: N802
            reached.append(self.headers.get("X-API-Key") or "<none>")
            self.send_response(200)
            self.end_headers()

    target, target_port = _server(Target)

    class Redirector(_QuietHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target_port}/capture")
            self.end_headers()

    source, source_port = _server(Redirector)
    try:
        client = HarnessClient(
            f"http://127.0.0.1:{source_port}",
            api_key="fx1k_synthetic",  # gitleaks:allow — probe literal
        )
        with pytest.raises(HarnessTransportError):
            client.commands()
        assert reached == []
    finally:
        source.shutdown()
        target.shutdown()


def test_credentialed_post_never_replays_across_a_307() -> None:
    """A 307 keeps the method+body — the most dangerous redirect shape for
    a credentialed POST; it must also refuse."""
    reached: list[str] = []

    class Target(_QuietHandler):
        def do_POST(self) -> None:  # noqa: N802
            reached.append(self.headers.get("X-API-Key") or "<none>")
            self.send_response(200)
            self.end_headers()

    target, target_port = _server(Target)

    class Redirector(_QuietHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(307)
            self.send_header("Location", f"http://127.0.0.1:{target_port}/capture")
            self.end_headers()

    source, source_port = _server(Redirector)
    try:
        client = HarnessClient(
            f"http://127.0.0.1:{source_port}",
            api_key="fx1k_synthetic",  # gitleaks:allow — probe literal
        )
        with pytest.raises((HarnessTransportError, RuntimeError)):
            client.complete(_MESSAGES)
        assert reached == []
    finally:
        source.shutdown()
        target.shutdown()


def test_userinfo_base_url_rejected_without_echoing_the_secret() -> None:
    with pytest.raises(ValueError) as denied:
        HarnessClient("http://user:fx1k_secret_pw@harness.invalid/")
    assert "fx1k_secret_pw" not in str(denied.value)
    assert "credentials" in str(denied.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://harness.invalid/path?api_key=fx1k_secret_pw",  # gitleaks:allow — probe literal
        "http://harness.invalid/path#fx1k_secret_pw",
        "http://harness.invalid/path;fx1k_secret_pw",
    ],
)
def test_extra_url_parts_rejected_without_echoing_the_secret(url: str) -> None:
    """Query/fragment/params on the base URL were silently dropped before —
    that is a silently-misparsed contract; refuse them instead, and never
    echo what may be a pasted credential."""
    with pytest.raises(ValueError) as denied:
        HarnessClient(url)
    assert "fx1k_secret_pw" not in str(denied.value)


def test_valid_base_url_keeps_its_path_prefix() -> None:
    def send(method: str, url: str, payload: Any, headers: dict[str, str], timeout_s: float):
        del method, payload, timeout_s, headers
        send.seen = url
        return (200, {}, b'{"items":[{"name":"alpha"}]}')

    send.seen = ""
    client = HarnessClient("http://harness.invalid/prefix", transport=send)
    assert client.commands() == ["alpha"]
    assert send.seen == "http://harness.invalid/prefix/harness/commands"


def _scripted_transport(*steps: Any) -> tuple[Any, list[tuple[str, str]]]:
    calls: list[tuple[str, str]] = []

    def send(method: str, url: str, payload: Any, headers: dict[str, str], timeout_s: float):
        del payload, headers, timeout_s
        calls.append((method, url))
        step = steps[len(calls) - 1]
        if isinstance(step, Exception):
            raise step
        return step

    return send, calls


def test_receipt_valid_flag_reads_headers_case_insensitively() -> None:
    """urllib preserves wire casing (``X-Fx1-Receipt-Valid``); a
    case-sensitive ``dict.get`` would silently read the live re-verify
    verdict as False on the production transport."""
    body = b'{"schema_tag":"receipt","kind":"x"}'
    transport, _ = _scripted_transport((200, {"X-Fx1-Receipt-Valid": "true"}, body))
    client = HarnessClient("http://harness.invalid", transport=transport)
    assert client.receipt("a" * 64).valid is True
    transport, _ = _scripted_transport((200, {"x-fx1-receipt-valid": "true"}, body))
    client = HarnessClient("http://harness.invalid", transport=transport)
    assert client.receipt("a" * 64).valid is True
    transport, _ = _scripted_transport((200, {}, body))
    client = HarnessClient("http://harness.invalid", transport=transport)
    assert client.receipt("a" * 64).valid is False


@pytest.mark.parametrize(
    "call, path",
    [
        (lambda c: c.key_get("k1?admin=true"), "/harness/keys/k1%3Fadmin%3Dtrue"),
        (lambda c: c.key_revoke("k1/x"), "/harness/keys/k1%2Fx"),
        (lambda c: c.file("f/../x"), "/v1/files/f%2F..%2Fx"),
        (lambda c: c.file_content("f?x=1"), "/v1/files/f%3Fx%3D1/content"),
        (lambda c: c.delete_file("f#x"), "/v1/files/f%23x"),
        (lambda c: c.upload_part("u?x=1", b"d"), "/v1/uploads/u%3Fx%3D1/parts"),
        (lambda c: c.upload_cancel("u/x"), "/v1/uploads/u%2Fx/cancel"),
        (lambda c: c.cancel_batch("b?x=1"), "/v1/batches/b%3Fx%3D1/cancel"),
        (lambda c: c.message_batch("mb/x"), "/v1/messages/batches/mb%2Fx"),
        (lambda c: c.receipt("aa/bb?cc"), "/receipts/aa%2Fbb%3Fcc"),
        (lambda c: c.job_status("j/x?y"), "/harness/jobs/j%2Fx%3Fy"),
    ],
)
def test_path_segment_params_are_single_segment_quoted(call: Any, path: str) -> None:
    """Caller-supplied ids interpolate into the request path — unquoted
    ``?``/``#``/``/`` would smuggle query params or sibling segments."""
    transport, calls = _scripted_transport((404, {}, b'{"detail":"nf"}'))
    client = HarnessClient("http://harness.invalid", transport=transport)
    with pytest.raises((KeyError, HarnessTransportError)):
        call(client)
    assert calls[0][1] == f"http://harness.invalid{path}"


def test_upload_file_rejects_header_injectable_filename() -> None:
    """``filename`` interpolates verbatim into a quoted multipart header
    field — quotes/backslash/CR-LF would break or inject the frame."""
    transport, calls = _scripted_transport((200, {}, b"{}"))
    client = HarnessClient("http://harness.invalid", transport=transport)
    for bad in ('evil".jsonl', "evil\r\nx: y", "evil\\name.jsonl", "evil\nname"):
        with pytest.raises(ValueError, match="filename"):
            client.upload_file(b"x", filename=bad)
    assert calls == []


def test_api_key_rides_the_header_and_nowhere_else() -> None:
    def send(method: str, url: str, payload: Any, headers: dict[str, str], timeout_s: float):
        del method, payload, timeout_s
        send.url = url
        send.headers = dict(headers)
        return (200, {}, b'{"items":[{"name":"alpha"}]}')

    client = HarnessClient(
        "http://harness.invalid",
        api_key="fx1k_synthetic",  # gitleaks:allow — probe literal
        transport=send,
    )
    client.commands()
    assert send.headers["X-API-Key"] == "fx1k_synthetic"
    assert "fx1k_synthetic" not in send.url


def test_anonymous_client_sends_no_auth_header() -> None:
    def send(method: str, url: str, payload: Any, headers: dict[str, str], timeout_s: float):
        del method, url, payload, timeout_s
        send.headers = dict(headers)
        return (200, {}, b'{"items":[{"name":"alpha"}]}')

    client = HarnessClient("http://harness.invalid", transport=send)
    client.commands()
    assert all(k.lower() != "x-api-key" for k in send.headers)


def test_error_envelope_bounds_the_echoed_body() -> None:
    """``_map_error`` may echo a bounded slice of the refusal body — never
    the whole thing (a hostile peer could stuff arbitrary bytes into the
    exception text)."""
    body = b'{"detail":"' + b"x" * 4096 + b'"}'
    transport, _ = _scripted_transport((500, {}, body))
    client = HarnessClient("http://harness.invalid", transport=transport)
    with pytest.raises(HarnessTransportError) as excinfo:
        client.commands()
    assert len(str(excinfo.value)) < 1024


def test_transport_receives_the_configured_timeout() -> None:
    def send(method: str, url: str, payload: Any, headers: dict[str, str], timeout_s: float):
        del method, url, payload, headers
        send.timeout = timeout_s
        return (200, {}, b'{"items":[{"name":"alpha"}]}')

    client = HarnessClient("http://harness.invalid", timeout_s=7.5, transport=send)
    client.commands()
    assert send.timeout == 7.5
