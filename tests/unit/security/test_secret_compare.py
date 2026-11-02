"""Credential comparisons use hmac.compare_digest, not ==."""

from __future__ import annotations

import re
from pathlib import Path

from fx1.secret_guard import secrets_equal

_ROOT = Path(__file__).resolve().parents[3]
_SCAN_ROOTS = ("src", "tests", "scripts")
_SKIP = {".git", ".venv", "venv", "__pycache__", ".dsh-24x7"}
_NAMES = (
    "api_key",
    "callback_secret",
    "api_secret",
    "webhook_secret",
    "secret_key",
    "client_secret",
    "access_token",
    "refresh_token",
    "signing_key",
    "password",
)
_OPS = "(?:" + "=" * 2 + "|!" + "=" + ")"
_SENTINELS = ("set", "unset", "[redacted]")


def _pattern() -> re.Pattern[str]:
    names = "|".join(_NAMES)
    return re.compile(
        "|".join(
            (
                rf"(?<![\w.])(?:{names})\s*{_OPS}",
                rf"{_OPS}\s*(?:{names})\b",
                rf"\.(?:{names})\s*{_OPS}",
                rf"\[\s*['\"](?:{names})['\"]\s*\]\s*{_OPS}",
                rf"\.get\(\s*['\"](?:{names})['\"]\s*\)\s*{_OPS}",
            )
        )
    )


def _strip_sentinels(line: str) -> str:
    cleaned = line
    for sentinel in _SENTINELS:
        for quote in ("'", '"'):
            literal = re.escape(quote + sentinel + quote)
            cleaned = re.sub(rf"{_OPS}\s*{literal}", " ", cleaned)
            cleaned = re.sub(rf"{literal}\s*{_OPS}", " ", cleaned)
    return cleaned


def test_secret_names_are_not_compared_with_equality() -> None:
    pattern = _pattern()
    hits: list[str] = []
    for folder in _SCAN_ROOTS:
        root = _ROOT / folder
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in _SKIP for part in path.parts):
                continue
            if path.name == "test_secret_compare.py":
                continue
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), 1):
                if pattern.search(_strip_sentinels(line)):
                    hits.append(f"{path.relative_to(_ROOT)}:{lineno}")
    assert hits == []


def test_secrets_equal_ignores_length_and_type() -> None:
    assert secrets_equal("alpha", "alpha")
    assert not secrets_equal("alpha", "alphabet")
    assert not secrets_equal("alpha", None)
    assert not secrets_equal(None, "alpha")
