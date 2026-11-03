"""SSRF, path confinement, shell, and local-engine argv guards."""

from __future__ import annotations

import json
import socket
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from fx1.interactive.shell import FxiShell
from fx1.serve.backends import engine_argv
from fx1.serve.client import HarnessTransportError, _urllib_transport
from fx1.serve.webhooks import check_callback_url
from fx1.webresearch.fetch import fetch_page
from quant_fund.research.ots_anchor import stamp_ots, verify_ots
from quant_fund.research.timestamp_anchor import stamp_timestamp, verify_timestamps
from quant_fund.utils.outbound import (
    PRIVATE_ALLOW_CIDRS_ENV,
    OutboundRefused,
    address_allowed,
    address_permitted,
    directory_child,
    is_public_ip,
    parse_allow_cidrs,
    public_addresses,
    read_http_redirects,
    repo_relative,
)


def _answer(ip: str, port: int = 80) -> list[tuple[object, ...]]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]


def _v6(ip: str, port: int = 80) -> list[tuple[object, ...]]:
    return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, port, 0, 0))]


def test_public_addresses_block_specials_and_mixed_answers() -> None:
    for host in ("192.88.99.1", "169.254.169.254", "100.64.0.1", "224.0.0.1", "::ffff:127.0.0.1"):
        with pytest.raises(OutboundRefused):
            public_addresses(host, 80)
    with pytest.raises(OutboundRefused):
        public_addresses("fec0::1", 80)
    assert public_addresses("8.8.8.8", 53) == ("8.8.8.8",)
    assert public_addresses("::ffff:8.8.8.8", 53) == ("8.8.8.8",)
    assert public_addresses("127.0.0.1", 9, allow_loopback=True) == ("127.0.0.1",)

    def mixed(host: str, port: int) -> list[tuple[object, ...]]:
        return _answer("8.8.8.8", port) + _answer("10.1.2.3", port)

    with pytest.raises(OutboundRefused, match="non-public"):
        public_addresses("evil.example", 80, resolve_fn=mixed)

    def missing(host: str, port: int) -> list[tuple[object, ...]]:
        raise OSError("no such host")

    with pytest.raises(OutboundRefused, match="cannot resolve"):
        public_addresses("ghost.invalid", 80, resolve_fn=missing)
    with pytest.raises(OutboundRefused):
        public_addresses("empty.example", 80, resolve_fn=lambda host, port: [])


def test_names_cannot_opt_into_private_answers() -> None:
    def loopback(host: str, port: int) -> list[tuple[object, ...]]:
        return _answer("127.0.0.1", port)

    with pytest.raises(OutboundRefused, match="non-public"):
        public_addresses("rebind.example", 80, allow_loopback=True, resolve_fn=loopback)
    assert public_addresses("localhost", 80, allow_loopback=True, resolve_fn=loopback) == (
        "127.0.0.1",
    )


def test_private_opt_in_still_blocks_link_local_and_documentation() -> None:
    import ipaddress

    assert address_allowed(ipaddress.ip_address("10.1.2.3"), allow_private=True)
    assert not address_allowed(ipaddress.ip_address("169.254.169.254"), allow_private=True)
    assert not address_allowed(ipaddress.ip_address("192.0.2.1"), allow_private=True)
    assert not address_allowed(ipaddress.ip_address("198.18.0.1"), allow_private=True)
    assert is_public_ip(ipaddress.ip_address("::ffff:8.8.8.8"))
    assert not is_public_ip(ipaddress.ip_address("fec0::1"))
    assert _v6("2001:4860:4860::8888")[0][0] == socket.AF_INET6


def test_webhook_literals_block_relay_and_metadata_even_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PRIVATE_ALLOW_CIDRS_ENV, raising=False)
    monkeypatch.delenv("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", raising=False)
    for url in ("http://192.88.99.1/hook", "http://[fec0::1]/hook", "http://169.254.169.254/"):
        with pytest.raises(ValueError):
            check_callback_url(url)
    monkeypatch.setenv(PRIVATE_ALLOW_CIDRS_ENV, "10.0.0.0/8")
    assert check_callback_url("http://10.1.2.3/hook") == "http://10.1.2.3/hook"
    with pytest.raises(ValueError):
        check_callback_url("http://192.168.1.1/hook")
    monkeypatch.setenv(PRIVATE_ALLOW_CIDRS_ENV, "0.0.0.0/0,169.254.0.0/16")
    with pytest.raises(ValueError):
        check_callback_url("http://169.254.169.254/latest/meta-data/")
    with pytest.raises(ValueError):
        check_callback_url("http://192.88.99.1/hook")
    with pytest.raises(ValueError):
        check_callback_url("http://192.0.2.1/hook")


def test_fetch_page_connects_only_to_the_pinned_address(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, int]] = []

    def resolve(host: str, port: int) -> list[tuple[object, ...]]:
        return _answer("93.184.216.34", port)

    def blocked(address: tuple[str, int], *args: object, **kwargs: object) -> None:
        seen.append(address)
        raise OSError("blocked")

    monkeypatch.setattr("quant_fund.utils.outbound.socket.create_connection", blocked)
    result = fetch_page("http://example.com/page", resolve_fn=resolve, timeout_s=1.0)
    assert seen == [("93.184.216.34", 80)]
    assert result.error


def test_fetch_page_refuses_redirect_to_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Handle:
        status = 302
        headers = {"Location": "http://169.254.169.254/latest/"}

        def read(self, amt: int | None = None) -> bytes:
            return b""

        def geturl(self) -> str:
            return "http://example.com/page"

    @contextmanager
    def fake_open(request: object, **kwargs: object) -> Iterator[_Handle]:
        yield _Handle()

    monkeypatch.setattr("fx1.webresearch.fetch.open_pinned", fake_open)
    result = fetch_page(
        "http://example.com/page",
        resolve_fn=lambda host, port: _answer("93.184.216.34", port),
        timeout_s=1.0,
    )
    assert "refused" in result.error
    assert result.ok is False


def test_paths_cannot_escape_the_repo(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "secret"
    outside.write_bytes(b"secret")
    (root / "inside").write_bytes(b"ok")
    assert repo_relative(root, "inside").read_bytes() == b"ok"
    with pytest.raises(OutboundRefused):
        repo_relative(root, "../secret")
    with pytest.raises(OutboundRefused):
        repo_relative(root, outside)
    link = root / "link"
    link.symlink_to(outside)
    with pytest.raises(OutboundRefused):
        repo_relative(root, "link")
    nested = root / "quality"
    nested.mkdir()
    assert directory_child(nested, "note.tsr").name == "note.tsr"
    with pytest.raises(OutboundRefused):
        directory_child(nested, "../secret")


def test_stamp_paths_do_not_touch_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    (root / "quality").mkdir(parents=True)
    outside = tmp_path / "secret"
    outside.write_bytes(b"secret")

    def boom(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("network")

    monkeypatch.setattr("quant_fund.research.timestamp_anchor.read_http", boom)
    monkeypatch.setattr("quant_fund.research.ots_anchor._exchange", boom)
    with pytest.raises(ValueError):
        stamp_timestamp("../secret", root=root)
    with pytest.raises(ValueError):
        stamp_ots("../secret", root=root, calendars=("https://x",))


def test_verify_rejects_manifest_escapes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    ts = root / "quality" / "timestamps"
    ts.mkdir(parents=True)
    (tmp_path / "secret.tsr").write_bytes(b"secret")
    (ts / "anchors.json").write_text(
        json.dumps(
            {
                "schema": "timestamp_anchors.v1",
                "anchors": {"../../secret.tsr": {"target": "x", "sha256": "ab"}},
            }
        )
    )
    with pytest.raises(ValueError):
        verify_timestamps(root)

    ots = ts / "ots"
    ots.mkdir()
    (ots / "ok.ots").write_bytes(b"not-a-token")
    (root / "quality" / "timestamps" / "ots_anchors.json").write_text(
        json.dumps(
            {
                "schema": "ots_anchors.v1",
                "anchors": {"ok.ots": {"target": "../../secret.tsr", "sha256": "ab" * 32}},
            }
        )
    )
    with pytest.raises(ValueError):
        verify_ots(root)


def test_engine_argv_keeps_the_checkpoint_path_as_one_argument() -> None:
    template = "$python -m fx1.serve.local_engine --checkpoint-dir $checkpoint_dir --port 9"
    argv = engine_argv(template, checkpoint_dir="/tmp/my checkpoint;rm", python="/usr/bin/python3")
    assert argv[0] == "/usr/bin/python3"
    assert "/tmp/my checkpoint;rm" in argv
    assert "rm" not in argv
    quoted = engine_argv('"$checkpoint_dir"', checkpoint_dir="/tmp/my checkpoint", python="python")
    assert quoted == ["/tmp/my checkpoint"]
    with pytest.raises(ValueError):
        engine_argv("$checkpoint_dir", checkpoint_dir="FX1_CHECKPOINT_DIR_TOKEN", python="python")


def test_shell_passthrough_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    def record(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        command = args[0] if args else ""
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(FxiShell, "_refresh_prompt", lambda self: None)
    monkeypatch.setattr("fx1.interactive.shell.subprocess.run", record)
    shell = FxiShell()
    monkeypatch.delenv("FX1_ALLOW_SHELL", raising=False)
    shell.do_shell("echo hi")
    assert calls == []
    monkeypatch.setenv("FX1_ALLOW_SHELL", "1")
    shell.do_shell("echo hi")
    assert calls and calls[0][0] == ("echo hi",)
    assert calls[0][1]["shell"] is True


def test_allowlist_parses_host_cidrs_and_hides_invalid_tokens() -> None:
    import ipaddress

    networks = parse_allow_cidrs(" 10.0.0.0/8, ,10.1.2.3, ::1/128 ")
    assert networks[0] == ipaddress.ip_network("10.0.0.0/8")
    assert networks[1] == ipaddress.ip_network("10.1.2.3/32")
    assert address_permitted(ipaddress.ip_address("10.1.2.3"), networks)
    assert not address_permitted(ipaddress.ip_address("192.168.0.1"), networks)
    assert address_permitted(ipaddress.ip_address("8.8.8.8"), ())
    wide = parse_allow_cidrs("0.0.0.0/0,::/0")
    for raw in ("169.254.169.254", "224.0.0.1", "192.0.2.1", "100.64.0.1", "::", "fe80::1"):
        assert not address_permitted(ipaddress.ip_address(raw), wide)
    assert address_permitted(ipaddress.ip_address("::1"), parse_allow_cidrs("::1/128"))
    with pytest.raises(OutboundRefused, match="allowlist CIDR is invalid") as raised:
        parse_allow_cidrs("not-a-cidr")
    assert raised.value.__cause__ is None


def test_harness_client_pins_resolved_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    real_getaddrinfo = socket.getaddrinfo
    dialed: list[tuple[str, int]] = []

    def resolve(host: str, port: int, *args: object, **kwargs: object) -> list[tuple[object, ...]]:
        if host == "public.example":
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.35", port)),
            ]
        if host == "metadata.example":
            return _answer("169.254.169.254", port)
        if host == "mixed.example":
            return [
                *_answer("93.184.216.34", port),
                *_answer("10.0.0.7", port),
            ]
        return real_getaddrinfo(host, port, *args, **kwargs)

    def blocked(address: tuple[str, int], *args: object, **kwargs: object) -> None:
        dialed.append(address)
        raise OSError("blocked")

    monkeypatch.setattr("quant_fund.utils.outbound.socket.getaddrinfo", resolve)
    monkeypatch.setattr("quant_fund.utils.outbound.socket.create_connection", blocked)

    for url in ("http://169.254.169.254/latest/", "http://192.88.99.1/"):
        dialed.clear()
        with pytest.raises(HarnessTransportError, match="destination address is not allowed"):
            _urllib_transport("GET", url, None, {}, 1.0)
        assert dialed == []

    with pytest.raises(HarnessTransportError, match="harness transport fault"):
        _urllib_transport("GET", "http://192.168.1.1/", None, {}, 1.0)
    assert dialed == [("192.168.1.1", 80)]

    dialed.clear()
    with pytest.raises(HarnessTransportError, match="harness transport fault"):
        _urllib_transport("GET", "https://public.example/health", None, {}, 1.0)
    assert dialed == [("93.184.216.34", 443)]

    dialed.clear()
    with pytest.raises(HarnessTransportError, match="non-public"):
        _urllib_transport("GET", "http://metadata.example/latest/", None, {}, 1.0)
    with pytest.raises(HarnessTransportError, match="non-public"):
        _urllib_transport("GET", "http://mixed.example/", None, {}, 1.0)
    assert dialed == []


class _Hop:
    def __init__(self, status: int, headers: Any, body: bytes, url: str) -> None:
        self.status = status
        self.reason = "test"
        self.headers = headers
        self._body = body
        self._url = url

    def read(self, amt: int | None = None) -> bytes:
        return self._body

    def geturl(self) -> str:
        return self._url

    def close(self) -> None:
        return None


def _redirect_plan(
    monkeypatch: pytest.MonkeyPatch,
    plan: dict[str, tuple[int, str | None, bytes]],
) -> list[tuple[str, str, bytes | None, dict[str, str]]]:
    import email.message
    import urllib.parse

    import quant_fund.utils.outbound as outbound_mod

    seen: list[tuple[str, str, bytes | None, dict[str, str]]] = []

    @contextmanager
    def fake_open(request: Any, **_kwargs: object) -> Iterator[object]:
        parsed = urllib.parse.urlparse(request.full_url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        public_addresses(parsed.hostname or "", port)
        seen.append(
            (request.full_url, request.get_method(), request.data, dict(request.header_items()))
        )
        status, location, body = plan[request.full_url]
        headers = email.message.Message()
        if location is not None:
            headers["Location"] = location
        yield _Hop(status, headers, body, request.full_url)

    monkeypatch.setattr(outbound_mod, "open_pinned", fake_open)
    return seen


def test_read_http_redirects_follows_a_public_hop(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    seen = _redirect_plan(
        monkeypatch,
        {
            "https://8.8.8.8/start": (302, "http://1.1.1.1/landed", b""),
            "http://1.1.1.1/landed": (200, None, b"landed"),
        },
    )
    body = read_http_redirects(urllib.request.Request("https://8.8.8.8/start"), timeout=1.0)
    assert body == b"landed"
    assert [item[0] for item in seen] == ["https://8.8.8.8/start", "http://1.1.1.1/landed"]


def test_read_http_redirects_refuses_a_metadata_hop(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    seen = _redirect_plan(
        monkeypatch,
        {"https://8.8.8.8/start": (302, "http://169.254.169.254/latest/meta-data", b"")},
    )
    with pytest.raises(OutboundRefused, match="not allowed"):
        read_http_redirects(urllib.request.Request("https://8.8.8.8/start"), timeout=1.0)
    assert [item[0] for item in seen] == ["https://8.8.8.8/start"]


def test_read_http_redirects_turns_303_into_get(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    seen = _redirect_plan(
        monkeypatch,
        {
            "https://8.8.8.8/post": (303, "http://1.1.1.1/done", b""),
            "http://1.1.1.1/done": (200, None, b"done"),
        },
    )
    request = urllib.request.Request(
        "https://8.8.8.8/post",
        data=b"{}",
        headers={"User-Agent": "dipcatcher-research/1.0", "Content-Type": "application/json"},
        method="POST",
    )
    assert read_http_redirects(request, timeout=1.0) == b"done"
    assert seen[1][1] == "GET"
    assert seen[1][2] is None
    forwarded = {name.lower() for name in seen[1][3]}
    assert "user-agent" in forwarded
    assert "content-type" not in forwarded
    assert "host" not in forwarded


def test_read_http_redirects_stops_on_a_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    seen = _redirect_plan(
        monkeypatch,
        {
            "https://8.8.8.8/loop": (302, "https://1.1.1.1/loop", b""),
            "https://1.1.1.1/loop": (302, "https://8.8.8.8/loop", b""),
        },
    )
    with pytest.raises(OutboundRefused, match="too many redirects"):
        read_http_redirects(
            urllib.request.Request("https://8.8.8.8/loop"), timeout=1.0, max_redirects=1
        )
    assert len(seen) == 2


def test_read_http_redirects_rejects_a_missing_location(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    _redirect_plan(monkeypatch, {"https://8.8.8.8/bare": (302, None, b"")})
    with pytest.raises(OutboundRefused, match="redirect missing Location"):
        read_http_redirects(urllib.request.Request("https://8.8.8.8/bare"), timeout=1.0)


def test_read_http_redirects_raises_http_error_for_a_final_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.error
    import urllib.request

    _redirect_plan(monkeypatch, {"https://8.8.8.8/missing": (404, None, b"nope")})
    with pytest.raises(urllib.error.HTTPError) as raised:
        read_http_redirects(urllib.request.Request("https://8.8.8.8/missing"), timeout=1.0)
    assert raised.value.code == 404
