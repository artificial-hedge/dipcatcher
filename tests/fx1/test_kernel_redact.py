"""Secret redaction: shapes, exact values, argv, and the things that must survive."""

from __future__ import annotations

import hashlib

import pytest

from fx1.interactive import redact
from fx1.interactive.redact import (
    REDACTED,
    SecretRegistry,
    contains_secret,
    fingerprint,
    is_sensitive_key,
    redact_argv,
    redact_text,
    redact_value,
)

KEY = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4"


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    redact.DEFAULT_REGISTRY.clear()
    yield
    redact.DEFAULT_REGISTRY.clear()


@pytest.mark.parametrize(
    "text",
    [
        f"MOONSHOT_API_KEY={KEY}",
        f'{{"api_key": "{KEY}"}}',
        f"Authorization: Bearer {KEY}",
        "authorization = Basic dXNlcjpwYXNzd29yZA==",
        "curl -H 'Authorization: Bearer abcdef0123456789ABCDEF' https://x",
        f"token in log: {KEY}",
        "https://alice:hunter2pw@host.example/v1/chat",
        "https://host.example/v1?api_key=abcdef123456&x=1",
        "client_secret: s3cr3t-value-123",
        "password=correcthorsebattery",
        "ghp_" + "A" * 36,
        "AKIAABCDEFGHIJKLMNOP",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk",
        "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg\n-----END PRIVATE KEY-----",
    ],
)
def test_credential_shapes_are_masked(text: str) -> None:
    out = redact_text(text)
    assert REDACTED in out
    assert contains_secret(text)
    for fragment in (KEY, "hunter2pw", "correcthorsebattery", "s3cr3t-value-123", "abcdef123456"):
        assert fragment not in out


def test_redaction_is_idempotent() -> None:
    once = redact_text(f"api_key={KEY} and Authorization: Bearer {KEY}")
    assert redact_text(once) == once
    assert not contains_secret(once)


def test_receipt_digests_and_token_counts_survive() -> None:
    digest = hashlib.sha256(b"receipt").hexdigest()
    text = (
        f"receipt_sha256={digest} total_tokens: 123456 max_tokens=100000 "
        f"prompt_tokens=4096 seed=1234567 hash {digest}"
    )
    assert redact_text(text) == text


def test_registered_exact_secret_is_removed_anywhere() -> None:
    registry = SecretRegistry()
    registry.register("plain-hunter-secret")
    out = redact_text("prefix plain-hunter-secret suffix", registry=registry)
    assert out == f"prefix {REDACTED} suffix"
    registry.register("tiny")  # below MIN_SECRET_LEN: ignored
    assert redact_text("tiny", registry=registry) == "tiny"


def test_default_registry_registers_longest_first() -> None:
    redact.register_secret("abcdefgh")
    redact.register_secret("abcdefghij")
    assert redact_text("x abcdefghij y") == f"x {REDACTED} y"


def test_redact_value_masks_sensitive_keys_and_recurses() -> None:
    payload = {
        "model": "fx1",
        "api_key": "whatever-it-is",
        "nested": [{"password": "p4ssw0rd!", "note": f"key is {KEY}"}],
        "empty_secret": "",
        "count": 3,
    }
    out = redact_value(payload)
    assert out["model"] == "fx1"
    assert out["api_key"] == REDACTED
    assert out["nested"][0]["password"] == REDACTED
    assert KEY not in out["nested"][0]["note"]
    assert out["empty_secret"] == ""
    assert out["count"] == 3


def test_redact_argv_masks_flag_values() -> None:
    argv = [
        "research",
        "--api-key",
        "supersecretvalue",
        "--token=abc12345678",
        "--config",
        "a.yaml",
    ]
    out = redact_argv(argv)
    assert out == ["research", "--api-key", REDACTED, f"--token={REDACTED}", "--config", "a.yaml"]


def test_fingerprint_is_presence_only() -> None:
    assert fingerprint(KEY) == f"...{KEY[-4:]}"
    assert KEY not in fingerprint(KEY)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("MOONSHOT_API_KEY", True),
        ("api_key", True),
        ("client_secret", True),
        ("access_token", True),
        ("password", True),
        ("max_tokens", False),
        ("total_tokens", False),
        ("receipt_sha256", False),
        ("model", False),
    ],
)
def test_is_sensitive_key(name: str, expected: bool) -> None:
    assert is_sensitive_key(name) is expected


def test_empty_and_non_secret_text_unchanged() -> None:
    assert redact_text("") == ""
    plain = "pinball loss 0.0123 on SYNTHETIC panel; receipt abc"
    assert redact_text(plain) == plain
