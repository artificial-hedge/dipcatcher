"""enforce_ship_gate=True refuses the fixture placeholder card."""

from __future__ import annotations

from pathlib import Path

import pytest

from fx1.modelcard import ModelCard
from fx1.serve.local_engine import LocalWeightsEngine

_CARD = Path(__file__).resolve().parents[2] / "artifacts" / "fx1_tiny_lm" / "modelcard.json"


def test_placeholder_card_is_not_ship_eligible() -> None:
    card = ModelCard.load(_CARD)
    assert card.eval_delta.is_placeholder
    assert card.ship_eligible is False


def test_enforce_ship_gate_refuses_placeholder_card(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "modelcard.json").write_text(_CARD.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a release candidate"):
        LocalWeightsEngine(ckpt, enforce_ship_gate=True)


def test_synthetic_fixture_does_not_bypass_a_failed_card(tmp_path: Path) -> None:
    from fx1.modelcard import EvalDelta

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    ModelCard(
        version="fx-1.v0.1",
        corpus_sha256="a" * 64,
        corpus_receipt_range="b..c",
        training_manifest_sha256="b" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.4,
            general_pass_rate_base=0.9,
            general_pass_rate_candidate=0.9,
            honesty_gate_candidate=False,
        ),
    ).save(ckpt / "modelcard.json")
    with pytest.raises(RuntimeError, match="SYNTHETIC fixture mode"):
        LocalWeightsEngine(ckpt, enforce_ship_gate=True, synthetic_fixture=True)
