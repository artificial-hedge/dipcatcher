"""Loopback URL policy: a zone id or percent suffix is not a literal address."""

from __future__ import annotations

import pytest

from fx1.serve.backends import _normalized_address, hosted_api_url_problem

_TOKEN = "zone-token"


@pytest.mark.parametrize(
    "url",
    [
        f"http://127.0.0.1%00.{_TOKEN}/v1/chat/completions",
        f"http://[::1%{_TOKEN}]/v1/chat/completions",
        f"http://[::1%25{_TOKEN}]/v1",
        "http://127.0.0.1%eth0/v1",
    ],
)
def test_percent_host_is_not_loopback_http(url: str) -> None:
    problem = hosted_api_url_problem(url)
    assert problem is not None
    assert "HTTPS" in problem or "loopback" in problem
    assert _TOKEN not in problem
    assert url not in problem
    assert "%" not in (problem or "")


def test_literal_loopback_http_stays_allowed() -> None:
    assert hosted_api_url_problem("http://127.0.0.1:9/v1/chat/completions") is None
    assert hosted_api_url_problem("http://[::1]/v1/chat/completions") is None
    assert hosted_api_url_problem("http://[::ffff:127.0.0.1]/v1") is None


def test_normalized_address_rejects_zone_and_keeps_mapped_loopback() -> None:
    with pytest.raises(ValueError, match="zone"):
        _normalized_address(f"127.0.0.1%00.{_TOKEN}")
    with pytest.raises(ValueError, match="zone"):
        _normalized_address(f"::1%{_TOKEN}")
    assert str(_normalized_address("127.0.0.1")) == "127.0.0.1"
    assert str(_normalized_address("::1")) == "::1"
    assert str(_normalized_address("::ffff:127.0.0.1")) == "127.0.0.1"
