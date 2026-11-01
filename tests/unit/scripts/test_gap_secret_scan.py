"""Tests for ``scripts/secret_scan.py``.

The scanner is a staged-diff credential gate. Its two properties that matter
are opposite to each other and both are pinned here:

1. **Fail closed** — a credential-shaped string in a staged line is reported
   and the commit is refused (exit 1).
2. **Do not manufacture failures** — this repo legitimately commits sha256
   digests and asset hashes, so a bare high-entropy blob is *not* a finding.
   Only credential *prefixes* and credential *context* (a key-shaped
   assignment) are.

The staged diff is faked, so nothing here depends on the real index.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SCAN_PATH = Path(__file__).resolve().parents[3] / "scripts" / "secret_scan.py"
_spec = importlib.util.spec_from_file_location("gap_secret_scan", _SCAN_PATH)
assert _spec and _spec.loader, "secret_scan.py must be importable"
scan_mod = importlib.util.module_from_spec(_spec)
sys.modules["gap_secret_scan"] = scan_mod
_spec.loader.exec_module(scan_mod)


def _staged(monkeypatch: pytest.MonkeyPatch, *lines: str) -> None:
    """Fake the staged-diff reader with the given added-line contents."""
    monkeypatch.setattr(
        scan_mod,
        "_staged_added_lines",
        lambda: [("some/file.py", index + 1, line) for index, line in enumerate(lines)],
    )


# --------------------------------------------------------------------------- #
# Provider patterns
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("AKIA0123456789ABCDEF", "aws_access_key"),
        ("ghp_0123456789abcdefghijklmnopqrstuvwxyz", "github_token"),
        ("github_pat_0123456789abcdefghijklmnop", "github_pat"),
        ("xoxb-0123456789-abcdefghij", "slack_token"),
        ("sk-0123456789abcdefghijklmn", "openai_key"),
        ("sk-proj-0123456789abcdefghijklmn", "openai_proj_key"),
        ("whsec_0123456789abcdefghij", "generic_webhook_secret"),
        ("sk_live_0123456789abcdefghij", "stripe_key"),
        ("-----BEGIN RSA PRIVATE KEY-----", "private_key_block"),
        ("-----BEGIN PRIVATE KEY-----", "private_key_block"),
        (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N",
            "jwt",
        ),
    ],
)
def test_provider_shapes_are_detected(
    monkeypatch: pytest.MonkeyPatch,
    content: str,
    expected: str,
) -> None:
    _staged(monkeypatch, content)
    findings = scan_mod.scan()
    assert len(findings) == 1
    assert expected in findings[0]


def test_a_credential_assignment_is_detected_by_its_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key-shaped assignment with a high-entropy value is a finding."""
    _staged(monkeypatch, 'api_key = "0123456789abcdefghijkl"')
    findings = scan_mod.scan()
    assert len(findings) == 1
    assert "credential_assignment(api_key)" in findings[0]


def test_a_bare_digest_is_not_a_finding(monkeypatch: pytest.MonkeyPatch) -> None:
    """The repo commits sha256 digests everywhere — a bare blob must stay clean."""
    _staged(
        monkeypatch, "digest = 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    )
    assert scan_mod.scan() == []


def test_a_placeholder_is_not_a_finding(monkeypatch: pytest.MonkeyPatch) -> None:
    _staged(monkeypatch, 'api_key = "your-api-key-here-placeholder"')
    assert scan_mod.scan() == []


def test_a_placeholder_hides_the_whole_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """A line marked as a placeholder is skipped wholesale."""
    _staged(monkeypatch, 'api_key = "AKIA0123456789ABCDEF"  # example')
    assert scan_mod.scan() == []


def test_the_provider_pattern_wins_over_the_assignment_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One finding per line, not two."""
    _staged(monkeypatch, 'api_key = "AKIA0123456789ABCDEF"')
    findings = scan_mod.scan()
    assert len(findings) == 1
    assert "aws_access_key" in findings[0]


# --------------------------------------------------------------------------- #
# Skipped paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path",
    ["artifacts/a.png", "artifacts/a.jpg", "data/a.parquet", "x.npz", "x.zip", "x.bin"],
)
def test_binary_paths_are_skipped(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    monkeypatch.setattr(
        scan_mod,
        "_staged_added_lines",
        lambda: [(path, 1, "AKIA0123456789ABCDEF")],
    )
    assert scan_mod.scan() == []


def test_the_scanner_itself_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scan_mod,
        "_staged_added_lines",
        lambda: [("scripts/secret_scan.py", 1, "AKIA0123456789ABCDEF")],
    )
    assert scan_mod.scan() == []


def test_a_baseline_file_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scan_mod,
        "_staged_added_lines",
        lambda: [("ci/app.secrets.baseline", 1, "AKIA0123456789ABCDEF")],
    )
    assert scan_mod.scan() == []


# --------------------------------------------------------------------------- #
# Diff parsing
# --------------------------------------------------------------------------- #


def test_added_lines_are_parsed_out_of_a_staged_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hunk header drives the line numbers reported in a finding."""
    diff = (
        b"diff --git a/app.py b/app.py\n"
        b"--- a/app.py\n"
        b"+++ b/app.py\n"
        b"@@ -10,0 +11,2 @@\n"
        b"+clean line\n"
        b"+AKIA0123456789ABCDEF\n"
    )
    monkeypatch.setattr(
        scan_mod.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, diff, b""),
    )
    assert scan_mod._staged_added_lines() == [
        ("app.py", 11, "clean line"),
        ("app.py", 12, "AKIA0123456789ABCDEF"),
    ]


def test_a_failed_git_call_scans_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scan_mod.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 128, b"", b"not a repo"),
    )
    assert scan_mod._staged_added_lines() == []
    assert scan_mod.scan() == []


def test_non_utf8_diff_bytes_are_decoded_tolerantly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A binary-looking hunk must not crash the scanner."""
    diff = b"+++ b/app.py\n@@ -0,0 +1,1 @@\n+\xff\xfe AKIA0123456789ABCDEF\n"
    monkeypatch.setattr(
        scan_mod.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, diff, b""),
    )
    hits = scan_mod._staged_added_lines()
    assert len(hits) == 1
    assert "AKIA0123456789ABCDEF" in hits[0][2]


# --------------------------------------------------------------------------- #
# Exit codes
# --------------------------------------------------------------------------- #


def test_main_returns_zero_for_a_clean_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    _staged(monkeypatch, "x = 1")
    assert scan_mod.main() == 0


def test_main_returns_one_when_a_secret_is_staged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _staged(monkeypatch, "AKIA0123456789ABCDEF")
    assert scan_mod.main() == 1
    assert "refusing commit" in capsys.readouterr().out
