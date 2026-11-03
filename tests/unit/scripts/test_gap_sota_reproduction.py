"""Tests for ``scripts/check_sota_reproduction.py``.

This gate answers one question: does a recomputed SOTA merge receipt still
equal its committed reference? Its design separates three kinds of difference
that are easy to conflate:

* **mismatch** — a real data difference. The gate fails closed (exit 1).
* **warning** — a difference in a field that legitimately changes between
  runs (implementation hash, loss-file name) or a float that is within the
  requested tolerance. The gate still passes.
* **normalised difference** — the same bytes spelled differently (absolute vs
  relative path, Windows vs POSIX separators). These are compared as data,
  not as spelling.

A loose claim here would be a false reproduction receipt, so every case is
pinned explicitly.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "check_sota_reproduction.py"
_spec = importlib.util.spec_from_file_location("gap_check_sota_reproduction", _SCRIPT_PATH)
assert _spec and _spec.loader, "check_sota_reproduction.py must be importable"
mod = importlib.util.module_from_spec(_spec)
sys.modules["gap_check_sota_reproduction"] = mod
_spec.loader.exec_module(mod)


def _compare(recomputed: object, reference: object, tol: float = 0.0):
    """Run the comparison and return ``(mismatches, warnings)``."""
    mismatches: list[str] = []
    warnings: list[str] = []
    mod._compare(recomputed, reference, "$", tol, mismatches, warnings)
    return mismatches, warnings


# --------------------------------------------------------------------------- #
# Exact reproduction
# --------------------------------------------------------------------------- #


def test_identical_payloads_reproduce_exactly() -> None:
    payload = {"a": 1, "b": "x", "c": [1, 2, 3], "d": {"e": True}}
    mismatches, warnings = _compare(payload, dict(payload))
    assert mismatches == []
    assert warnings == []


def test_a_changed_value_is_a_mismatch() -> None:
    mismatches, _ = _compare({"a": 2}, {"a": 1})
    assert len(mismatches) == 1
    assert "a" in mismatches[0]


def test_a_changed_length_is_a_mismatch() -> None:
    mismatches, _ = _compare([1, 2, 3], [1, 2])
    assert len(mismatches) == 1
    assert "len" in mismatches[0]


def test_a_key_missing_from_the_recomputed_payload_is_a_mismatch() -> None:
    mismatches, _ = _compare({"a": 1}, {"a": 1, "b": 2})
    assert len(mismatches) == 1
    assert "missing in recomputed" in mismatches[0]


def test_a_key_only_in_the_recomputed_payload_is_a_mismatch() -> None:
    mismatches, _ = _compare({"a": 1, "b": 2}, {"a": 1})
    assert len(mismatches) == 1
    assert "missing in reference" in mismatches[0]


def test_a_type_change_is_a_mismatch() -> None:
    mismatches, _ = _compare({"a": "1"}, {"a": 1})
    assert len(mismatches) == 1
    assert "type" in mismatches[0]


# --------------------------------------------------------------------------- #
# Volatile and warn-only fields
# --------------------------------------------------------------------------- #


def test_a_volatile_timestamp_is_ignored_entirely() -> None:
    """``created_at`` differs between runs and is not a data failure."""
    mismatches, warnings = _compare(
        {"created_at": "2024-01-01", "a": 1}, {"created_at": "2025-01-01", "a": 1}
    )
    assert mismatches == []
    assert warnings == []


def test_a_warn_only_field_is_reported_but_does_not_fail() -> None:
    mismatches, warnings = _compare(
        {"implementation_sha256": "abc", "a": 1},
        {"implementation_sha256": "def", "a": 1},
    )
    assert mismatches == []
    assert len(warnings) == 1
    assert "implementation_sha256" in warnings[0]


def test_a_warn_only_field_that_matches_produces_no_warning() -> None:
    mismatches, warnings = _compare({"losses_file": "a.npz"}, {"losses_file": "a.npz"})
    assert mismatches == []
    assert warnings == []


# --------------------------------------------------------------------------- #
# Path normalisation
# --------------------------------------------------------------------------- #


def test_a_basename_keyed_map_compares_by_basename() -> None:
    """The same parts spelled with different directories are the same parts."""
    mismatches, _ = _compare(
        {"source_parts_sha256": {"/abs/dir/a.parquet": "aaa"}},
        {"source_parts_sha256": {"rel/a.parquet": "aaa"}},
    )
    assert mismatches == []


def test_a_basename_keyed_map_with_different_digests_is_a_mismatch() -> None:
    mismatches, _ = _compare(
        {"source_parts_sha256": {"/abs/dir/a.parquet": "aaa"}},
        {"source_parts_sha256": {"rel/a.parquet": "bbb"}},
    )
    assert mismatches != []


def test_a_parts_list_compares_as_sorted_basenames() -> None:
    mismatches, _ = _compare(
        {"parts": ["/x/b.parquet", "/x/a.parquet"]},
        {"parts": ["a.parquet", "b.parquet"]},
    )
    assert mismatches == []


def test_a_parts_list_with_a_different_member_is_a_mismatch() -> None:
    mismatches, _ = _compare(
        {"parts": ["/x/b.parquet", "/x/c.parquet"]},
        {"parts": ["a.parquet", "b.parquet"]},
    )
    assert mismatches != []


def test_windows_separators_normalise_to_the_same_basename() -> None:
    mismatches, _ = _compare(
        {"source_parts_sha256": {"C:\\data\\a.parquet": "aaa"}},
        {"source_parts_sha256": {"a.parquet": "aaa"}},
    )
    assert mismatches == []


# --------------------------------------------------------------------------- #
# Floats
# --------------------------------------------------------------------------- #


def test_floats_compare_exactly_by_default() -> None:
    mismatches, _ = _compare({"a": 1.0000001}, {"a": 1.0})
    assert mismatches != []


def test_a_float_inside_the_tolerance_is_a_warning() -> None:
    mismatches, warnings = _compare({"a": 1.0001}, {"a": 1.0}, tol=0.001)
    assert mismatches == []
    assert len(warnings) == 1
    assert "rel_tol" in warnings[0]


def test_a_float_outside_the_tolerance_is_a_mismatch() -> None:
    mismatches, _ = _compare({"a": 1.1}, {"a": 1.0}, tol=0.001)
    assert mismatches != []


def test_two_nans_are_equal() -> None:
    """NaN != NaN in Python, but two NaN fields must not read as a mismatch."""
    mismatches, _ = _compare({"a": float("nan")}, {"a": float("nan")})
    assert mismatches == []


def test_an_infinite_float_is_compared_by_value() -> None:
    mismatches, _ = _compare({"a": float("inf")}, {"a": float("inf")})
    assert mismatches == []


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(monkeypatch: pytest.MonkeyPatch, recomputed: Path, reference: Path, *extra: str) -> int:
    """Drive ``main()`` through argv, the way the CI job invokes it."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_sota_reproduction.py",
            "--recomputed",
            str(recomputed),
            "--reference",
            str(reference),
            *extra,
        ],
    )
    return mod.main()


def test_main_passes_on_an_exact_reproduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recomputed = _write(tmp_path / "recomputed.json", {"a": 1, "b": "x"})
    reference = _write(tmp_path / "reference.json", {"a": 1, "b": "x"})
    assert _run(monkeypatch, recomputed, reference) == 0
    assert "reproduction verified" in capsys.readouterr().out


def test_main_fails_closed_on_a_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recomputed = _write(tmp_path / "recomputed.json", {"a": 2})
    reference = _write(tmp_path / "reference.json", {"a": 1})
    assert _run(monkeypatch, recomputed, reference) == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_passes_a_warning_only_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recomputed = _write(tmp_path / "recomputed.json", {"implementation_sha256": "abc", "a": 1})
    reference = _write(tmp_path / "reference.json", {"implementation_sha256": "def", "a": 1})
    assert _run(monkeypatch, recomputed, reference) == 0
    out = capsys.readouterr().out
    assert "warning:" in out
    assert "reproduction verified" in out


def test_main_honours_the_float_tolerance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recomputed = _write(tmp_path / "recomputed.json", {"a": 1.0001})
    reference = _write(tmp_path / "reference.json", {"a": 1.0})
    assert _run(monkeypatch, recomputed, reference, "--float-rel-tol", "0.01") == 0
    assert "reproduction verified" in capsys.readouterr().out
