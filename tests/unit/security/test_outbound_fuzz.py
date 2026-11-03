"""Property tests for outbound address classification.

Obfuscated literals must be decided before DNS. Tunnel forms stay blocked
even when an allowlist contains them.
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
from urllib.request import Request

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.utils.outbound import (
    OutboundRefused,
    address_allowed,
    address_permitted,
    is_public_ip,
    open_pinned,
    public_addresses,
)

_WIDE = (
    ipaddress.ip_network("::/0"),
    ipaddress.ip_network("0.0.0.0/0"),
    ipaddress.ip_network("2002::/16"),
    ipaddress.ip_network("2001::/32"),
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("::ffff:0:0/96"),
)


def _boom(host: str, port: int) -> list[object]:
    raise AssertionError(f"resolver used for {host}")


def _sixtofour(address: ipaddress.IPv4Address) -> ipaddress.IPv6Address:
    return ipaddress.IPv6Address((0x2002 << 112) | (int(address) << 80))


def _nat64(address: ipaddress.IPv4Address) -> ipaddress.IPv6Address:
    return ipaddress.IPv6Address(int(ipaddress.IPv6Address("64:ff9b::")) | int(address))


def _teredo(server: ipaddress.IPv4Address, client: ipaddress.IPv4Address) -> ipaddress.IPv6Address:
    packed = (0x20010000 << 96) | (int(server) << 64) | (int(client) ^ 0xFFFFFFFF)
    return ipaddress.IPv6Address(packed)


def _fullwidth(text: str) -> str:
    chars: list[str] = []
    for char in text:
        if char.isdigit():
            chars.append(chr(0xFF10 + int(char)))
        elif char == ".":
            chars.append("．")
        else:
            chars.append(char)
    return "".join(chars)


def _round_trip(text: str, address: ipaddress.IPv4Address) -> str | None:
    try:
        packed = socket.inet_aton(text)
    except OSError:
        return None
    if int.from_bytes(packed, "big") != int(address):
        return None
    return text


def _encodings(address: ipaddress.IPv4Address) -> list[str]:
    number = int(address)
    dotted = str(address)
    candidates = [
        dotted,
        str(number),
        hex(number),
        "0" + format(number, "o"),
        f"{dotted}.",
        _fullwidth(dotted),
        f"::ffff:{dotted}",
        f"[::ffff:{dotted}]",
    ]
    kept: list[str] = []
    for host in candidates:
        if host in {
            dotted,
            f"{dotted}.",
            _fullwidth(dotted),
            f"::ffff:{dotted}",
            f"[::ffff:{dotted}]",
        }:
            kept.append(host)
            continue
        if _round_trip(host, address) is not None:
            kept.append(host)
    return kept


def _expect_literal(host: str, address: ipaddress.IPv4Address) -> None:
    if address_allowed(address):
        assert public_addresses(host, 80, resolve_fn=_boom) == (str(address),)
        return
    with pytest.raises(OutboundRefused):
        public_addresses(host, 80, resolve_fn=_boom)


@settings(max_examples=40, deadline=None)
@given(number=st.integers(min_value=0, max_value=2**32 - 1))
def test_obfuscated_ipv4_is_classified_without_dns(number: int) -> None:
    address = ipaddress.IPv4Address(number)
    for host in _encodings(address):
        _expect_literal(host, address)


@settings(max_examples=25, deadline=None)
@given(
    number=st.integers(min_value=0, max_value=2**32 - 1),
    server=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_tunnel_forms_stay_blocked_under_a_wide_allowlist(number: int, server: int) -> None:
    address = ipaddress.IPv4Address(number)
    forms = (
        _sixtofour(address),
        _nat64(address),
        _teredo(ipaddress.IPv4Address(server), address),
        ipaddress.IPv6Address(f"::ffff:0:{address}"),
    )
    for form in forms:
        assert not is_public_ip(form)
        assert not address_allowed(form, allow_loopback=True, allow_private=True)
        assert not address_permitted(form, _WIDE)
        with pytest.raises(OutboundRefused):
            public_addresses(str(form), 80, resolve_fn=_boom)


def test_known_literal_bypasses_never_reach_dns() -> None:
    blocked = (
        "2130706433",
        "0x7f000001",
        "017700000001",
        "127.1",
        "0177.0.0.1",
        "0x7f.0.0.1",
        "127.0.0.1.",
        "0",
        "127.000.000.001",
        _fullwidth("127.0.0.1"),
        "127。0。0。1",
        "::ffff:127.0.0.1",
        "[::ffff:169.254.169.254]",
        "2002:a9fe:a9fe::",
        "64:ff9b::a9fe:a9fe",
        "64:ff9b::7f00:1",
    )
    for host in blocked:
        with pytest.raises(OutboundRefused):
            public_addresses(host, 80, resolve_fn=_boom)
    assert public_addresses(_fullwidth("8.8.8.8"), 53, resolve_fn=_boom) == ("8.8.8.8",)
    assert public_addresses("134744072", 53, resolve_fn=_boom) == ("8.8.8.8",)
    assert public_addresses("8.8.8.8.", 53, resolve_fn=_boom) == ("8.8.8.8",)


def test_homoglyph_localhost_cannot_opt_into_loopback() -> None:
    host = "l\u043ecalh\u043est"

    def loopback(_name: str, port: int) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    with pytest.raises(OutboundRefused, match="non-public"):
        public_addresses(host, 80, allow_loopback=True, resolve_fn=loopback)
    assert public_addresses("localhost.", 80, allow_loopback=True, resolve_fn=loopback) == (
        "127.0.0.1",
    )


def test_dns_answer_cannot_change_between_resolve_and_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    seen: list[tuple[str, int]] = []

    def resolve(host: str, port: int) -> list[tuple[object, ...]]:
        calls.append(host)
        if len(calls) == 1:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))]

    def connect(address: tuple[str, int], *_args: object, **_kwargs: object) -> None:
        seen.append(address)
        raise OSError("stopped")

    monkeypatch.setattr("quant_fund.utils.outbound.socket.create_connection", connect)
    with (
        pytest.raises(OSError, match="stopped"),
        open_pinned(
            Request("http://rebind.example/path"),
            timeout=1.0,
            resolve_fn=resolve,
        ),
    ):
        pass
    assert calls == ["rebind.example"]
    assert seen == [("93.184.216.34", 80)]


def test_http_failure_still_closes_the_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []

    class _Conn:
        def request(self, *_args: object, **_kwargs: object) -> None:
            raise http.client.RemoteDisconnected("eof")

        def close(self) -> None:
            closed.append("closed")

    monkeypatch.setattr("quant_fund.utils.outbound.pinned_connection", lambda *_a, **_k: _Conn())
    with (
        pytest.raises(http.client.RemoteDisconnected),
        open_pinned(Request("http://8.8.8.8/"), timeout=1.0),
    ):
        pass
    assert closed == ["closed"]
