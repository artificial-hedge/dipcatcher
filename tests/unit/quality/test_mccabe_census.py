"""Tests for the census-backed McCabe pin mode.

``scripts/check_mccabe_ratchet.py --allow-with-census <census.json>`` exists to
pin never-pinned pre-existing debt while leaving every existing pin untouched.
Its safety property is that it can only ever *widen* the baseline: it must never
raise a recorded value, and it must refuse a census that was hand-edited or that
no longer matches the tree.

These tests use injected fake scans rather than a live ruff run so the
behavioural contract is pinned without a 13k-module scan in CI.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MCCABE = runpy.run_path(str(ROOT / "scripts" / "check_mccabe_ratchet.py"))

build_census = MCCABE["build_census"]
census_rows = MCCABE["census_rows"]
load_census = MCCABE["load_census"]
write_baseline_from_census = MCCABE["write_baseline_from_census"]
_format_baseline = MCCABE["_format_baseline"]
_census_digest = MCCABE["_census_digest"]

CENSUS_SCHEMA = "dipcatcher.mccabe_census/1"


def _finding(file: str, function: str, line: int, complexity: int) -> dict[str, object]:
    return {"file": file, "function": function, "line": line, "complexity": complexity}


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the script's ROOT/BASELINE at a temp tree."""
    baseline = tmp_path / "quality" / "mccabe_baseline.txt"
    baseline.parent.mkdir(parents=True)
    monkeypatch.setitem(MCCABE["write_baseline_from_census"].__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(MCCABE["write_baseline_from_census"].__globals__, "BASELINE", baseline)
    return tmp_path


def _census_file(tmp_path: Path, findings: list[dict[str, object]]) -> Path:
    census = build_census(findings, ceiling=74, scope=["src"])
    path = tmp_path / "census.json"
    path.write_text(json.dumps(census), encoding="utf-8")
    return path


def _pin_scan(monkeypatch: pytest.MonkeyPatch, findings: list[dict[str, object]]) -> None:
    """Make the fresh scan inside write_baseline_from_census return `findings`."""
    monkeypatch.setitem(
        write_baseline_from_census.__globals__,
        "_ruff_findings",
        lambda _paths: findings,
    )


def test_build_census_is_self_describing_and_deterministic() -> None:
    findings = [
        _finding("src/a.py", "big", 10, 182),
        _finding("src/b.py", "mid", 3, 75),
        _finding("src/c.py", "small", 1, 74),
    ]
    census = build_census(findings, ceiling=74, scope=["src"])
    assert census["schema"] == CENSUS_SCHEMA
    assert census["ceiling"] == 74
    assert census["scope"] == ["src"]
    assert str(census["tool"]).startswith("ruff")
    assert census["violation_count"] == 2
    # Only strictly-above-ceiling functions are violations; 74 is not a violation.
    assert [v["function"] for v in census["violations"]] == ["big", "mid"]
    for violation in census["violations"]:
        assert set(violation) == {"file", "function", "line", "complexity"}
    # Byte-reproducible: generated_at is excluded from the digest.
    again = build_census(findings, ceiling=74, scope=["src"])
    assert again["digest"] == census["digest"]
    assert json.dumps(_without_timestamp(again), sort_keys=True) == json.dumps(
        _without_timestamp(census), sort_keys=True
    )


def _without_timestamp(census: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in census.items() if k != "generated_at"}


def test_census_digest_covers_payload_so_hand_edits_are_detected() -> None:
    findings = [_finding("src/a.py", "big", 10, 182)]
    census = build_census(findings, ceiling=74, scope=["src"])
    assert census["digest"] == _census_digest(census["violations"], 74, ["src"], census["tool"])
    # Changing the complexity without recomputing the digest breaks the match.
    assert (
        _census_digest([_finding("src/a.py", "big", 10, 181)], 74, ["src"], census["tool"])
        != (census["digest"])
    )


def test_load_census_rejects_wrong_schema_and_missing_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_schema = tmp_path / "a.json"
    bad_schema.write_text(json.dumps({"schema": "other/1"}))
    with pytest.raises(SystemExit):
        load_census(bad_schema)
    assert "dipcatcher.mccabe_census/1" in capsys.readouterr().err

    incomplete = tmp_path / "b.json"
    incomplete.write_text(json.dumps({"schema": CENSUS_SCHEMA, "ceiling": 74}))
    with pytest.raises(SystemExit):
        load_census(incomplete)
    assert "digest" in capsys.readouterr().err

    not_json = tmp_path / "c.json"
    not_json.write_text("{nope")
    with pytest.raises(SystemExit):
        load_census(not_json)


def test_census_rows_keeps_max_on_duplicate_function() -> None:
    rows = census_rows(
        {
            "violations": [
                _finding("src/a.py", "f", 1, 90),
                _finding("src/a.py", "f", 2, 182),
            ]
        }
    )
    assert rows == {("src/a.py", "f"): 182}


def test_allow_with_census_pins_new_keys_and_never_raises(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The core safety property: new debt is pinned, existing pins never move up."""
    baseline = sandbox / "quality" / "mccabe_baseline.txt"
    baseline.write_text(
        _format_baseline({("src/quant_fund/x.py", "regressed"): 10}),
        encoding="utf-8",
    )
    findings = [
        # Existing pin, now MORE complex -> must stay at 10, not become 13.
        _finding("src/quant_fund/x.py", "regressed", 10, 13),
        # Brand-new function never pinned -> gets its measured value.
        _finding("src/quant_fund/y.py", "fresh", 4, 21),
    ]
    census_path = _census_file(sandbox, findings)
    _pin_scan(monkeypatch, findings)

    assert write_baseline_from_census(census_path) == 0

    written = baseline.read_text(encoding="utf-8")
    rows = MCCABE["_entries"](baseline)
    assert rows[("src/quant_fund/x.py", "regressed")] == 10, "pin was raised!"
    assert rows[("src/quant_fund/y.py", "fresh")] == 21
    # Canonical sorted form is preserved.
    assert written == _format_baseline(rows)
    # The regression is reported as still-failing, not silently blessed.
    err = capsys.readouterr().err
    assert "complexity 13 > baseline 10" in err
    assert "regressed" in err


def test_allow_with_census_refuses_hand_edited_census(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = sandbox / "quality" / "mccabe_baseline.txt"
    findings = [_finding("src/quant_fund/y.py", "fresh", 4, 182)]
    census_path = _census_file(sandbox, findings)
    tampered = json.loads(census_path.read_text())
    tampered["violations"][0]["complexity"] = 999
    census_path.write_text(json.dumps(tampered), encoding="utf-8")
    _pin_scan(monkeypatch, findings)

    assert write_baseline_from_census(census_path) == 1
    assert "digest mismatch" in capsys.readouterr().err
    assert not baseline.exists(), "baseline must not be written on refusal"


def test_allow_with_census_refuses_stale_census_against_moved_tree(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A census that no longer matches the tree must fail closed."""
    baseline = sandbox / "quality" / "mccabe_baseline.txt"
    at_census_time = [_finding("src/quant_fund/y.py", "fresh", 4, 182)]
    census_path = _census_file(sandbox, at_census_time)
    # The tree grew a new offender after the census was taken.
    tree_now = [*at_census_time, _finding("src/quant_fund/z.py", "appeared", 7, 200)]
    _pin_scan(monkeypatch, tree_now)

    assert write_baseline_from_census(census_path) == 1
    err = capsys.readouterr().err
    assert "tree has moved" in err
    assert not baseline.exists(), "baseline must not be written on refusal"


def test_main_rejects_write_and_allow_with_census_together(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main = MCCABE["main"]
    census = _census_file(sandbox, [_finding("src/quant_fund/y.py", "fresh", 4, 21)])
    assert main(["--write", "--allow-with-census", str(census)]) == 2
    assert "mutually exclusive" in capsys.readouterr().err
