"""Live redaction: real receipt bytes, escaped secrets, and a time-boxed corpus."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fx1.interactive.redact import REDACTED, SecretRegistry, redact_text, redact_value

_RECEIPTS = (
    Path("/workspace/receipts/api_audit.json"),
    Path("/workspace/receipts/cli_audit.json"),
    Path("/workspace/receipts/kill_audit.json"),
    Path("/workspace/receipts/asof_audit.json"),
)
_KEY = "sk-" + "a1B2c3D4e5F6g7H8i9J0"


def test_real_receipts_are_not_mangled() -> None:
    digest = "ab" * 32
    git_sha = "c" * 40
    ulid = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    sample = "\n".join(
        [
            f"receipt_sha256={digest}",
            f"git {git_sha}",
            f"id {uuid.UUID(int=0)}",
            f"ulid {ulid}",
            "path /workspace/receipts/kill_audit.json",
            "n 12345678901234567890",
            "total_tokens: 4096",
        ]
    )
    assert redact_text(sample) == sample
    for path in _RECEIPTS:
        text = path.read_text(encoding="utf-8")
        assert redact_text(text) == text
        assert digest not in text or digest in redact_text(text)


def test_nested_json_and_unicode_escapes_are_scrubbed() -> None:
    payload = {"outer": {"note": f"key is {_KEY}", "api_key": "plain-value-here", "n": 3}}
    redacted = redact_value(payload)
    assert _KEY not in json.dumps(redacted)
    assert redacted["outer"]["api_key"] == REDACTED
    assert redacted["outer"]["n"] == 3
    escaped = "prefix \\u0073" + _KEY[1:] + " suffix"
    out = redact_text(escaped)
    assert _KEY[3:] not in out
    assert "\\u0073k-" not in out
    registry = SecretRegistry()
    registry.register("plain-hunter-secret")
    hidden = "see \\u0070lain-hunter-secret now"
    assert "plain-hunter-secret" not in redact_text(hidden, registry=registry)
    assert redact_text(out) == out


def test_url_userinfo_query_bearer_pem_and_assignments() -> None:
    url = "https://alice:sec@ret@host.example/v1?api_key=abcdef123456&x=1"
    out = redact_text(url)
    assert "sec" not in out and "ret" not in out and "abcdef123456" not in out
    assert redact_text(out) == out
    bearer = "Authorization: Bearer\n" + _KEY
    assert _KEY not in redact_text(bearer)
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEvQIBADANBg\n-----END RSA PRIVATE KEY-----"
    assert "MIIEvQIBADANBg" not in redact_text(pem)
    assignment = f"export AWS_SECRET_ACCESS_KEY={_KEY}"
    assert _KEY not in redact_text(assignment)
    split = "sk-" + ("a" * 10) + "\n" + ("b" * 10)
    assert "aaaaaaaaaa" not in redact_text(split)
    assert "bbbbbbbbbb" not in redact_text(split)


def test_five_megabytes_stays_linear() -> None:
    blob = "a" * 5_000_000
    started = time.perf_counter()
    assert redact_text(blob) == blob
    assert time.perf_counter() - started < 5.0
    hostile = ("b" * 2_000_000) + f"api_key={_KEY}" + ("c" * 2_000_000)
    started = time.perf_counter()
    scrubbed = redact_text(hostile)
    assert time.perf_counter() - started < 5.0
    assert _KEY not in scrubbed
    assert redact_text(scrubbed) == scrubbed
