"""Endpoint normalization: origin, prefix, and completion route."""

from __future__ import annotations

import pytest

from fx1.interactive.endpoint import EndpointError, EndpointSpec, normalize_endpoint
from fx1.serve.backends import MOONSHOT_API_URL

_SECRET = "s3cret-token"
_HOST = "https://h.example"


def _assert_spec(
    raw: str,
    *,
    origin: str,
    prefix: str,
    completion: str,
    host: str,
    route_added: bool,
) -> EndpointSpec:
    spec = normalize_endpoint(raw)
    assert spec == EndpointSpec(origin, prefix, completion, host, route_added)
    again = normalize_endpoint(spec.completion_url)
    assert again.origin == spec.origin
    assert again.api_prefix == spec.api_prefix
    assert again.completion_url == spec.completion_url
    assert again.host == spec.host
    assert again == normalize_endpoint(again.completion_url)
    assert again.describe() == f"POST {completion}"
    return spec


@pytest.mark.parametrize(
    ("raw", "origin", "prefix", "completion", "host", "route_added"),
    [
        (
            "https://API.Example.TEST",
            "https://api.example.test",
            "/v1",
            "https://api.example.test/v1/chat/completions",
            "api.example.test",
            True,
        ),
        (
            "https://api.example.test/",
            "https://api.example.test",
            "/v1",
            "https://api.example.test/v1/chat/completions",
            "api.example.test",
            True,
        ),
        (
            "https://api.example.test",
            "https://api.example.test",
            "/v1",
            "https://api.example.test/v1/chat/completions",
            "api.example.test",
            True,
        ),
        (
            "HTTPS://API.Example.TEST/v1",
            "https://api.example.test",
            "/v1",
            "https://api.example.test/v1/chat/completions",
            "api.example.test",
            True,
        ),
        (
            "https://api.example.test/v1/",
            "https://api.example.test",
            "/v1",
            "https://api.example.test/v1/chat/completions",
            "api.example.test",
            True,
        ),
        (
            "https://api.example.test/v1/chat/completions",
            "https://api.example.test",
            "/v1",
            "https://api.example.test/v1/chat/completions",
            "api.example.test",
            False,
        ),
        (
            "https://api.example.test/v1/chat/completions/",
            "https://api.example.test",
            "/v1",
            "https://api.example.test/v1/chat/completions",
            "api.example.test",
            False,
        ),
        (
            "https://api.example.test/openai/v1",
            "https://api.example.test",
            "/openai/v1",
            "https://api.example.test/openai/v1/chat/completions",
            "api.example.test",
            True,
        ),
        (
            "https://api.example.test/openai/v1/chat/completions",
            "https://api.example.test",
            "/openai/v1",
            "https://api.example.test/openai/v1/chat/completions",
            "api.example.test",
            False,
        ),
        (
            "https://api.example.test/chat/completions",
            "https://api.example.test",
            "",
            "https://api.example.test/chat/completions",
            "api.example.test",
            False,
        ),
        (
            "HTTPS://API.Example.TEST/V1/chat/completions",
            "https://api.example.test",
            "/V1",
            "https://api.example.test/V1/chat/completions",
            "api.example.test",
            False,
        ),
        (
            "https://api.example.test//v1",
            "https://api.example.test",
            "/v1",
            "https://api.example.test/v1/chat/completions",
            "api.example.test",
            True,
        ),
        (
            "https://api.example.test/v1//chat///completions/",
            "https://api.example.test",
            "/v1",
            "https://api.example.test/v1/chat/completions",
            "api.example.test",
            False,
        ),
        (
            "  \t https://api.example.test/v1 \t ",
            "https://api.example.test",
            "/v1",
            "https://api.example.test/v1/chat/completions",
            "api.example.test",
            True,
        ),
        (
            "https://api.\u200bexample.test/v1",
            "https://api.example.test",
            "/v1",
            "https://api.example.test/v1/chat/completions",
            "api.example.test",
            True,
        ),
        (
            "https://api.example.test/v 1",
            "https://api.example.test",
            "/v1",
            "https://api.example.test/v1/chat/completions",
            "api.example.test",
            True,
        ),
        (
            "http://127.0.0.1:8080",
            "http://127.0.0.1:8080",
            "/v1",
            "http://127.0.0.1:8080/v1/chat/completions",
            "127.0.0.1",
            True,
        ),
        (
            "http://127.0.0.1:8080/v1/chat/completions",
            "http://127.0.0.1:8080",
            "/v1",
            "http://127.0.0.1:8080/v1/chat/completions",
            "127.0.0.1",
            False,
        ),
        (
            "http://[::1]/v1",
            "http://[::1]",
            "/v1",
            "http://[::1]/v1/chat/completions",
            "::1",
            True,
        ),
        (
            "http://[::1]:8080",
            "http://[::1]:8080",
            "/v1",
            "http://[::1]:8080/v1/chat/completions",
            "::1",
            True,
        ),
        (
            "http://[::FFFF:127.0.0.1]/v1",
            "http://[::ffff:127.0.0.1]",
            "/v1",
            "http://[::ffff:127.0.0.1]/v1/chat/completions",
            "::ffff:127.0.0.1",
            True,
        ),
        (
            "https://h.example:8443/openai/V1",
            "https://h.example:8443",
            "/openai/V1",
            "https://h.example:8443/openai/V1/chat/completions",
            "h.example",
            True,
        ),
        (
            "https://h.example:443/v1/chat/completions",
            "https://h.example:443",
            "/v1",
            "https://h.example:443/v1/chat/completions",
            "h.example",
            False,
        ),
        (
            MOONSHOT_API_URL,
            "https://api.moonshot.ai",
            "/v1",
            MOONSHOT_API_URL,
            "api.moonshot.ai",
            False,
        ),
        (
            "https://h.example/v1/chat/completions/extra",
            "https://h.example",
            "/v1/chat/completions/extra",
            "https://h.example/v1/chat/completions/extra/chat/completions",
            "h.example",
            True,
        ),
    ],
)
def test_normalize_success(
    raw: str,
    origin: str,
    prefix: str,
    completion: str,
    host: str,
    route_added: bool,
) -> None:
    spec = _assert_spec(
        raw,
        origin=origin,
        prefix=prefix,
        completion=completion,
        host=host,
        route_added=route_added,
    )
    assert "artificialhedge" not in spec.completion_url


def test_route_added_is_per_call() -> None:
    spec = normalize_endpoint("HTTPS://API.Example.TEST")
    again = normalize_endpoint(spec.completion_url)
    assert spec.route_added is True
    assert again.route_added is False
    assert again == normalize_endpoint(again.completion_url)
    assert spec.describe() == "POST https://api.example.test/v1/chat/completions"


def test_completion_url_at_length_limit_is_idempotent() -> None:
    origin = "https://h.example"
    route = "/chat/completions"
    prefix = "/" + ("b" * (2048 - len(origin) - len(route) - 1))
    completion = origin + prefix + route
    assert len(completion) == 2048
    spec = normalize_endpoint(completion)
    assert spec.completion_url == completion
    assert normalize_endpoint(spec.completion_url) == spec


def test_expansion_past_length_limit_is_rejected() -> None:
    base = "https://h.example/"
    raw = base + ("c" * (2048 - len(base)))
    assert len(raw) == 2048
    with pytest.raises(EndpointError, match="2048") as exc:
        normalize_endpoint(raw)
    assert raw not in str(exc.value)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("\u200b\ufeff", "empty"),
        ("https://h.example/v1\n", "newline"),
        ("https://h.example/v1\r", "newline"),
        ("https://h.example/\x00v1", "NUL"),
        ("ftp://h.example/v1", "http"),
        ("file:///tmp/x", "http"),
        ("not-a-url", "http"),
        (f"https://user:{_SECRET}@h.example/v1", "userinfo"),
        (f"https://h.example/v1?api_key={_SECRET}", "query"),
        (f"https://h.example/v1#{_SECRET}", "fragment"),
        ("https://h.example/v1?", "query"),
        ("https://h.example/v1#", "fragment"),
        (f"https://h.example/v1;drop={_SECRET}", "params"),
        ("http://models.example.test/v1", "HTTPS"),
        ("http://localhost/v1", "HTTPS"),
        (f"http://evil.example/{_SECRET}", "HTTPS"),
        ("https://h.example:0/v1", "port"),
        ("https://h.example:65536/v1", "port"),
        ("https://xn--e28h.com/v1", "punycode"),
        ("https://exämple.com/v1", "ASCII"),
        ("https://", "http"),
        ("https:///v1", "http"),
        ("a" * 2049, "2048"),
    ],
)
def test_normalize_rejects(raw: str, match: str) -> None:
    with pytest.raises(EndpointError, match=match) as exc:
        normalize_endpoint(raw)
    message = str(exc.value)
    assert _SECRET not in message
    assert "user:" not in message
    if len(raw) >= 8:
        assert raw not in message
    assert len(message) < 200


def test_endpoint_error_is_value_error() -> None:
    assert issubclass(EndpointError, ValueError)


def test_spec_is_frozen() -> None:
    spec = normalize_endpoint(_HOST)
    with pytest.raises(AttributeError):
        spec.host = "other"  # type: ignore[misc]
