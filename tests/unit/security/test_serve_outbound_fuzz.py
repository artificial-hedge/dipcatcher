"""Per-caller property tests for the shared outbound selector.

BYOK and webhooks must classify obfuscated literals before DNS, refuse
transitional embeddings under a wide allowlist, and dial the vetted address.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fx1.serve import backends, webhooks
from quant_fund.utils.outbound import PRIVATE_ALLOW_CIDRS_ENV, is_public_ip
from tests.unit.security.test_outbound_fuzz import (
    _encodings,
    _nat64,
    _sixtofour,
    _teredo,
)

_WIDE = "0.0.0.0/0,::/0,2002::/16,2001::/32,64:ff9b::/96,::ffff:0:0/96"


def _refuse_dns(*_args: Any, **_kwargs: Any) -> list[object]:
    raise AssertionError("resolver used")


@settings(max_examples=20, deadline=None)
@given(number=st.integers(min_value=0, max_value=2**32 - 1))
def test_byok_obfuscated_literals_never_reach_dns(number: int) -> None:
    address = ipaddress.IPv4Address(number)
    with (
        patch.dict(os.environ, {PRIVATE_ALLOW_CIDRS_ENV: ""}, clear=False),
        patch("socket.getaddrinfo", side_effect=_refuse_dns),
    ):
        for host in _encodings(address):
            if is_public_ip(address):
                assert backends._byok_resolved_addresses(host, 443, allow_private=False) == (
                    str(address),
                )
            else:
                with pytest.raises(ValueError, match="prohibited network address"):
                    backends._byok_resolved_addresses(host, 443, allow_private=False)


@settings(max_examples=15, deadline=None)
@given(number=st.integers(min_value=0, max_value=2**32 - 1))
def test_byok_transitional_prefixes_stay_blocked(number: int) -> None:
    address = ipaddress.IPv4Address(number)
    hosts = (
        str(_sixtofour(address)),
        str(_nat64(address)),
        str(_teredo(ipaddress.IPv4Address("198.51.100.10"), address)),
    )
    with (
        patch.dict(os.environ, {PRIVATE_ALLOW_CIDRS_ENV: _WIDE}, clear=False),
        patch("socket.getaddrinfo", side_effect=_refuse_dns),
    ):
        for host in hosts:
            with pytest.raises(ValueError, match="prohibited network address"):
                backends._byok_resolved_addresses(host, 443, allow_private=True)


def test_byok_connect_pins_the_first_vetted_address(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    seen: list[str] = []

    def resolve(host: str, port: int, *_args: Any, **_kwargs: Any) -> list[object]:
        del host
        calls["n"] += 1
        if calls["n"] == 1:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))]

    class Connection:
        def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
            del host, port, timeout
            seen.append(address)

        def request(self, *_args: Any, **_kwargs: Any) -> None:
            raise OSError("stopped")

        def close(self) -> None:
            return None

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(backends, "_ByokPinnedHTTPConnection", Connection)
    with pytest.raises(urllib.error.URLError):
        backends._open_byok_pinned(
            urllib.request.Request("http://provider.example/v1"),
            timeout_s=1,
            allow_private=False,
        )
    assert calls["n"] == 1
    assert seen == ["93.184.216.34"]


def test_byok_homoglyph_is_not_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    def resolve(host: str, port: int, *_args: Any, **_kwargs: Any) -> list[object]:
        assert host == "l\u043ecalh\u043est"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    with pytest.raises(ValueError, match="prohibited network address"):
        backends._byok_resolved_addresses("l\u043ecalh\u043est", 80, allow_private=False)
    with pytest.raises(ValueError, match="prohibited network address"):
        backends._byok_resolved_addresses(
            "l\u043ecalh\u043est", 80, allow_private=False, local_literals=True
        )


@settings(max_examples=20, deadline=None)
@given(number=st.integers(min_value=0, max_value=2**32 - 1))
def test_webhook_obfuscated_literals_never_reach_dns(number: int) -> None:
    address = ipaddress.IPv4Address(number)
    with (
        patch.dict(os.environ, {PRIVATE_ALLOW_CIDRS_ENV: ""}, clear=False),
        patch("socket.getaddrinfo", side_effect=_refuse_dns),
    ):
        for host in _encodings(address):
            if is_public_ip(address):
                assert webhooks._resolved_addresses(host, 443) == (str(address),)
                webhooks._validate_literal_host(host)
            else:
                with pytest.raises(ValueError, match="private or special-use"):
                    webhooks._resolved_addresses(host, 443)
                with pytest.raises(ValueError, match="private or special-use"):
                    webhooks._validate_literal_host(host)


@settings(max_examples=15, deadline=None)
@given(number=st.integers(min_value=0, max_value=2**32 - 1))
def test_webhook_transitional_prefixes_stay_blocked(number: int) -> None:
    address = ipaddress.IPv4Address(number)
    hosts = (
        str(_sixtofour(address)),
        str(_nat64(address)),
        str(_teredo(ipaddress.IPv4Address("198.51.100.10"), address)),
    )
    with (
        patch.dict(os.environ, {PRIVATE_ALLOW_CIDRS_ENV: _WIDE}, clear=False),
        patch("socket.getaddrinfo", side_effect=_refuse_dns),
    ):
        for host in hosts:
            with pytest.raises(ValueError, match="private or special-use"):
                webhooks.check_callback_url(f"http://[{host}]/hook")


def test_webhook_delivery_pins_the_first_vetted_address() -> None:
    calls = {"n": 0}

    def resolve(host: str, port: int, *_args: Any, **_kwargs: Any) -> list[object]:
        del host
        calls["n"] += 1
        if calls["n"] == 1:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))]

    with (
        patch("fx1.serve.webhooks.socket.getaddrinfo", side_effect=resolve),
        patch("fx1.serve.webhooks._post_once", return_value=204) as post,
    ):
        delivered, error, attempts = webhooks.deliver_signed(
            "https://callback.example/hook",
            None,
            b"{}",
            max_attempts=1,
            backoff_s=0,
        )
    assert delivered is True
    assert error is None
    assert attempts == 1
    assert calls["n"] == 1
    assert post.call_args.args[1] == "93.184.216.34"


def test_webhook_homoglyph_does_not_opt_into_loopback() -> None:
    host = "l\u043ecalh\u043est"
    with patch("socket.getaddrinfo", side_effect=_refuse_dns):
        assert webhooks.check_callback_url(f"http://{host}/hook") == f"http://{host}/hook"

    def resolve(name: str, port: int, *_args: Any, **_kwargs: Any) -> list[object]:
        assert name == host
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    with (
        patch("fx1.serve.webhooks.socket.getaddrinfo", side_effect=resolve),
        patch("fx1.serve.webhooks._post_once") as post,
    ):
        delivered, error, attempts = webhooks.deliver_signed(
            f"http://{host}/hook",
            None,
            b"{}",
            max_attempts=1,
            backoff_s=0,
        )
    assert delivered is False
    assert error is not None and "private or special-use" in error
    assert attempts == 1
    post.assert_not_called()


def test_webhook_trailing_dot_public_literal_skips_dns() -> None:
    with patch("socket.getaddrinfo", side_effect=_refuse_dns):
        assert webhooks.check_callback_url("http://8.8.8.8./hook") == "http://8.8.8.8./hook"
        assert webhooks._resolved_addresses("8.8.8.8.", 80) == ("8.8.8.8",)
        parsed = urllib.parse.urlparse("http://8.8.8.8./hook")
        assert parsed.hostname is not None
