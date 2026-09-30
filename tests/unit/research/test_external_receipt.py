"""External-submission receipt verification (referee path), fail-closed.

Every fixture below is SYNTHETIC - a correctness test of the verifier, never
market evidence. A passing result is provenance only: it stays
``proof_status="not_proof"`` and never becomes a performance or live claim.
"""

from __future__ import annotations

import hashlib
import json
import runpy
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS
from quant_fund.research.external_receipt import verify_receipt

pytestmark = pytest.mark.synthetic

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_SCRIPT = REPO_ROOT / "scripts" / "verify_external_receipt.py"

VINTAGES = ("2026-01-05", "2026-02-02")
BASE_ROWS: dict[str, dict[str, str]] = {
    "2026-01-05": {
        "2025-12-29": "14.10",
        "2025-12-30": "14.62",
        "2025-12-31": ".",
    },
    "2026-02-02": {
        "2025-12-29": "14.10",
        # Restatement between vintages: the later vintage revises this print.
        "2025-12-30": "14.71",
        "2025-12-31": "15.03",
        "2026-01-02": "15.44",
    },
}


def _value_column(vintage: str) -> str:
    return f"VIXCLS_{vintage.replace('-', '')}"


def _write_snapshot(root: Path, vintage: str, rows: Mapping[str, str]) -> Path:
    """Write one SYNTHETIC vintage CSV exactly as an external submitter would."""
    lines = [f"observation_date,{_value_column(vintage)}"]
    lines.extend(f"{date},{value}" for date, value in sorted(rows.items()))
    path = root / f"synthetic_vixcls_{vintage.replace('-', '')}.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _snapshot_entry(path: Path, vintage: str, *, rel_to: Path | None = None) -> dict[str, Any]:
    stored = path.relative_to(rel_to).as_posix() if rel_to is not None else path.name
    return {
        "vintage_date": vintage,
        "path": stored,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pit_status": "candidate_only",
        "proof_status": "not_proof",
    }


def _write_receipt(
    root: Path,
    snapshots: list[dict[str, Any]],
    *,
    series: str = "VIXCLS",
    name: str = "external_receipt.json",
) -> Path:
    payload = {"series": series, "synthetic": True, "snapshots": snapshots}
    path = root / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def submission(tmp_path: Path) -> Iterator[dict[str, Any]]:
    """A valid two-vintage SYNTHETIC submission under ``tmp_path/'submission'``."""
    root = tmp_path / "submission"
    root.mkdir()
    paths = {vintage: _write_snapshot(root, vintage, BASE_ROWS[vintage]) for vintage in VINTAGES}
    receipt = _write_receipt(
        root,
        [_snapshot_entry(paths[vintage], vintage, rel_to=root) for vintage in VINTAGES],
    )
    yield {"root": root, "receipt": receipt, "paths": dict(paths)}


def test_valid_external_receipt_verifies(submission: dict[str, Any]) -> None:
    result = verify_receipt(submission["receipt"])
    assert result["ok"] is True
    assert result["snapshot_count"] == 2
    assert result["hashes_verified"] == 2
    # ``ok`` is integrity, never proof; the label is carried into the result.
    assert result["proof_status"] == "not_proof"
    pairs = result["revision_pairs"]
    assert [pair["older_vintage"] for pair in pairs] == ["2026-01-05"]
    assert [pair["newer_vintage"] for pair in pairs] == ["2026-02-02"]
    assert pairs[0]["common_observations"] == 3
    assert pairs[0]["value_changes"] == 1
    # "." -> "15.03" is an availability change, not a restated value.
    assert pairs[0]["availability_changes"] == 1
    assert pairs[0]["revisions"] == [
        {
            "observation_date": "2025-12-30",
            "older_value": "14.62",
            "newer_value": "14.71",
        },
        {
            "observation_date": "2025-12-31",
            "older_value": None,
            "newer_value": "15.03",
        },
    ]


def test_base_dir_resolves_relative_snapshot_paths(tmp_path: Path) -> None:
    """Receipt paths are relative to ``--base-dir`` so a clone can relocate."""
    root = tmp_path / "submission"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    paths = {
        vintage: _write_snapshot(elsewhere, vintage, BASE_ROWS[vintage]) for vintage in VINTAGES
    }
    receipt = _write_receipt(
        root, [_snapshot_entry(paths[vintage], vintage) for vintage in VINTAGES]
    )
    # Without base_dir the snapshots are not beside the receipt: fail closed.
    with pytest.raises(ValueError, match="snapshot file does not exist"):
        verify_receipt(receipt)
    result = verify_receipt(receipt, base_dir=elsewhere)
    assert result["ok"] is True
    assert result["hashes_verified"] == 2


def test_tampered_snapshot_bytes_fail(submission: dict[str, Any]) -> None:
    path: Path = submission["paths"]["2026-01-05"]
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("14.10", "13.99"), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_receipt(submission["receipt"])


def test_tampered_receipt_field_fails(submission: dict[str, Any]) -> None:
    receipt: Path = submission["receipt"]
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["series"] = "VIX9D"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="series must be VIXCLS"):
        verify_receipt(receipt)


def test_tampered_hash_field_fails(submission: dict[str, Any]) -> None:
    receipt: Path = submission["receipt"]
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["snapshots"][0]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_receipt(receipt)


def test_missing_referenced_snapshot_fails_closed(submission: dict[str, Any]) -> None:
    path: Path = submission["paths"]["2026-02-02"]
    path.unlink()
    with pytest.raises(ValueError, match="snapshot file does not exist"):
        verify_receipt(submission["receipt"])


@pytest.mark.parametrize("value", ["nan", "NaN", "-nan", "inf", "Infinity", "-inf"])
def test_non_finite_metric_fails_closed(tmp_path: Path, value: str) -> None:
    """A NaN/Inf print is not a usable observation: reject, do not coerce."""
    rows = {"2025-12-29": "14.10", "2025-12-30": value}
    path = _write_snapshot(tmp_path, VINTAGES[0], rows)
    receipt = _write_receipt(tmp_path, [_snapshot_entry(path, VINTAGES[0])])
    with pytest.raises(ValueError, match="non-finite value"):
        verify_receipt(receipt)


def test_non_numeric_metric_fails_closed(tmp_path: Path) -> None:
    path = _write_snapshot(tmp_path, VINTAGES[0], {"2025-12-29": "not-a-number"})
    receipt = _write_receipt(tmp_path, [_snapshot_entry(path, VINTAGES[0])])
    with pytest.raises(ValueError, match="non-numeric value"):
        verify_receipt(receipt)


def test_malformed_observation_rows_fail_closed(tmp_path: Path) -> None:
    path = _write_snapshot(tmp_path, VINTAGES[0], {"2025-12-29": "14.10", "2025-12-30": "14.62"})
    path.write_text(
        path.read_text(encoding="utf-8").replace("2025-12-30", "2025-12-30\n2025-12-30"),
        encoding="utf-8",
        newline="\n",
    )
    receipt = _write_receipt(tmp_path, [_snapshot_entry(path, VINTAGES[0])])
    with pytest.raises(ValueError, match="duplicate observation date"):
        verify_receipt(receipt)


def test_mismatched_value_column_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "synthetic_vixcls_20260105.csv"
    path.write_text("observation_date,VIXCLS\n2025-12-29,14.10\n", encoding="utf-8", newline="\n")
    receipt = _write_receipt(tmp_path, [_snapshot_entry(path, VINTAGES[0])])
    with pytest.raises(ValueError, match="unexpected VIXCLS value column"):
        verify_receipt(receipt)


def test_snapshot_must_stay_candidate_only_and_not_proof(tmp_path: Path) -> None:
    """Honesty contract: an external snapshot cannot arrive pre-promoted."""
    path = _write_snapshot(tmp_path, VINTAGES[0], {"2025-12-29": "14.10"})
    entry = _snapshot_entry(path, VINTAGES[0])
    entry["pit_status"] = "proven"
    receipt = _write_receipt(tmp_path, [entry])
    with pytest.raises(ValueError, match="pit_status must be candidate_only"):
        verify_receipt(receipt)

    entry = _snapshot_entry(path, VINTAGES[0])
    entry["proof_status"] = "proof"
    receipt = _write_receipt(tmp_path, [entry], name="external_receipt2.json")
    with pytest.raises(ValueError, match="proof_status must be not_proof"):
        verify_receipt(receipt)


def test_duplicate_vintage_fails_closed(tmp_path: Path) -> None:
    first = _write_snapshot(tmp_path, VINTAGES[0], {"2025-12-29": "14.10"})
    second = tmp_path / "duplicate.csv"
    second.write_bytes(first.read_bytes())
    entries = [_snapshot_entry(first, VINTAGES[0]), _snapshot_entry(second, VINTAGES[0])]
    receipt = _write_receipt(tmp_path, entries)
    with pytest.raises(ValueError, match="duplicate vintage date"):
        verify_receipt(receipt)


def test_empty_snapshot_list_fails_closed(tmp_path: Path) -> None:
    receipt = _write_receipt(tmp_path, [])
    with pytest.raises(ValueError, match="non-empty list"):
        verify_receipt(receipt)


def test_malformed_snapshot_entry_fails_closed(tmp_path: Path) -> None:
    receipt = _write_receipt(tmp_path, [{"vintage_date": VINTAGES[0]}])
    with pytest.raises(ValueError, match="non-empty vintage_date, path, and sha256"):
        verify_receipt(receipt)


def _walk_keys(node: Any) -> Iterator[str]:
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield str(key)
            yield from _walk_keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_keys(item)


def test_result_never_carries_a_live_performance_claim(submission: dict[str, Any]) -> None:
    result = verify_receipt(submission["receipt"])
    assert result["proof_status"] == "not_proof"
    keys = {key.lower() for key in _walk_keys(result)}
    for token in FORBIDDEN_RESEARCH_METRIC_KEYS:
        assert not any(token in key for key in keys), token
    assert not any("live" in key or "trading" in key for key in keys)
    assert set(result) == {
        "ok",
        "proof_status",
        "snapshot_count",
        "hashes_verified",
        "revision_pairs",
    }
    # The serialized verdict carries the not-proof label end to end.
    assert '"proof_status": "not_proof"' in json.dumps(result, sort_keys=True)


def test_cli_script_prints_the_verdict(
    submission: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI at ``scripts/verify_external_receipt.py`` must import and run."""
    assert CLI_SCRIPT.is_file()
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_external_receipt.py",
            str(submission["receipt"]),
            "--base-dir",
            str(submission["root"]),
        ],
    )
    runpy.run_path(str(CLI_SCRIPT), run_name="__main__")
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["snapshot_count"] == 2
    assert payload["proof_status"] == "not_proof"
