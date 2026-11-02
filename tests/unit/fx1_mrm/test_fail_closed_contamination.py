"""Contamination evidence must not last-win its way to a clean dossier."""

from __future__ import annotations

import json
from pathlib import Path

from fx1.modelcard import EvalDelta, ModelCard
from fx1.mrm import compile_dossier


def _card(tmp_path: Path) -> Path:
    path = tmp_path / "modelcard.json"
    ModelCard(
        version="fx-1.v0.1",
        corpus_sha256="a" * 64,
        corpus_receipt_range="b..c",
        training_manifest_sha256="b" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.7,
            general_pass_rate_base=0.9,
            general_pass_rate_candidate=0.9,
            honesty_gate_candidate=True,
        ),
    ).save(path)
    return path


def _complete_artifacts(tmp_path: Path, validation: Path) -> dict[str, Path]:
    artifacts = {"validation": validation}
    for activity in ("development", "implementation", "monitoring"):
        path = tmp_path / f"{activity}.json"
        path.write_text("{}", encoding="utf-8")
        artifacts[activity] = path
    return artifacts


def test_duplicate_overall_flagged_cannot_certify_clean(tmp_path: Path) -> None:
    """Renaming the file must not let a repeated flag last-win to clean."""
    report = tmp_path / "audit.json"
    report.write_text(
        '{"overall_flagged": true, "overall_flagged": false}',
        encoding="utf-8",
    )
    dossier = compile_dossier(
        modelcard_path=_card(tmp_path),
        artifacts=_complete_artifacts(tmp_path, report),
        out_path=tmp_path / "dossier.json",
    )
    assert dossier.complete is True
    assert dossier.contamination_flagged is True
    assert dossier.ship_eligible is False


def test_unique_clean_flag_still_allows_a_complete_dossier(tmp_path: Path) -> None:
    report = tmp_path / "audit.json"
    report.write_text(json.dumps({"overall_flagged": False}), encoding="utf-8")
    dossier = compile_dossier(
        modelcard_path=_card(tmp_path),
        artifacts=_complete_artifacts(tmp_path, report),
        out_path=tmp_path / "dossier.json",
    )
    assert dossier.contamination_flagged is False
    assert dossier.ship_eligible is True


def test_declared_undecodable_contamination_report_cannot_certify_clean(tmp_path: Path) -> None:
    report = tmp_path / "notes.json"
    report.write_bytes(b"\xff")
    dossier = compile_dossier(
        modelcard_path=_card(tmp_path),
        artifacts={"contamination_report": report},
        out_path=tmp_path / "dossier.json",
    )
    assert dossier.contamination_flagged is True
    assert dossier.ship_eligible is False
