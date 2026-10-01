"""Recursive corpus-scan coverage: a claim receipt under ``receipts/<sub>/``
is corpus evidence — every scanner (evidence audit, suite health, lattice,
proofcore receipt listing) must see it, while quarantined subdirs stay out.
"""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.proofcore.ci import receipt_paths
from quant_fund.research.evidence_audit import audit_receipts_dir
from quant_fund.research.receipt_lattice import receipt_lattice
from quant_fund.utils.receipt import is_quarantined, verified_corpus_files


def _receipt(inputs: str, score: float) -> dict:
    return {
        "schema": "demo.v1",
        "inputs_sha256": inputs,
        "results": [{"head": "a", "pinball": score}],
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
    }


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def test_verified_corpus_files_recurse_and_skip_quarantine(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", _receipt("x", 0.1))
    _write(tmp_path / "drills" / "b.json", _receipt("x", 0.2))
    _write(tmp_path / "legacy-unsealed" / "old.json", _receipt("x", 0.3))
    _write(tmp_path / "legacy-unsealed" / "deep" / "older.json", _receipt("x", 0.4))
    files = verified_corpus_files(tmp_path)
    rels = [p.relative_to(tmp_path).as_posix() for p in files]
    assert rels == ["a.json", "drills/b.json"]


def test_is_quarantined_only_top_level_subdir(tmp_path: Path) -> None:
    assert is_quarantined(tmp_path / "legacy-unsealed" / "f.json", tmp_path)
    assert not is_quarantined(tmp_path / "drills" / "legacy-unsealed" / "f.json", tmp_path)
    # outside the root is never audited — treated as quarantined
    assert is_quarantined(Path("/elsewhere/f.json"), tmp_path)


def test_evidence_audit_audits_subdir_receipts(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", _receipt("x", 0.1))
    _write(tmp_path / "drills" / "b.json", _receipt("x", 0.2))
    _write(tmp_path / "legacy-unsealed" / "old.json", _receipt("x", 0.3))
    rows = audit_receipts_dir(tmp_path)
    files = {r["file"] for r in rows}
    assert files == {"a.json", "drills/b.json"}


def test_lattice_edges_subdir_claims(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", _receipt("in-a", 0.42))
    _write(tmp_path / "drills" / "b.json", _receipt("in-a", 0.99))
    out = receipt_lattice(tmp_path)
    assert out["n_receipts"] == 2
    assert out["verdict"] == "inconsistent"
    group = next(g for g in out["groups"] if g["verdict"] == "inconsistent")
    assert group["files"] == ["a.json", "drills/b.json"]


def test_receipt_paths_recurse_minus_quarantine(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", _receipt("x", 0.1))
    _write(tmp_path / "legacy-unsealed" / "old.json", _receipt("x", 0.3))
    nested = tmp_path / "runs"
    _write(nested / "b.json", _receipt("x", 0.2))
    names = [p.relative_to(tmp_path).as_posix() for p in receipt_paths(tmp_path)]
    assert names == ["a.json", "runs/b.json"]


def test_suite_health_counts_subdir_receipts(tmp_path: Path) -> None:
    from quant_fund.research.suite_health import suite_health

    _write(tmp_path / "a.json", _receipt("x", 0.1))
    _write(tmp_path / "drills" / "b.json", _receipt("x", 0.2))
    _write(tmp_path / "legacy-unsealed" / "old.json", _receipt("x", 0.3))
    frame, receipt = suite_health(tmp_path)
    assert frame.height == 2
    assert receipt["n_receipts"] == 2
    assert set(receipt["params"]["input_labels"]) == {"a.json", "drills/b.json"}


def test_verified_corpus_files_respects_pattern(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", _receipt("x", 0.1))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "notes.md").write_text("# notes\n")
    json_files = verified_corpus_files(tmp_path, pattern="*.json")
    md_files = verified_corpus_files(tmp_path, pattern="*.md")
    assert [p.name for p in json_files] == ["a.json"]
    assert [p.name for p in md_files] == ["notes.md"]
