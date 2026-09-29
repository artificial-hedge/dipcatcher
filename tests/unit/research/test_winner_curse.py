"""winner_curse: selection-bias correction on tournament argmin."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.winner_curse import (
    audit_from_streams,
    winner_curse_audit,
)


def _tied_streams(n_heads: int, n_obs: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {f"h{k}": rng.normal(0.0, 1.0, n_obs) for k in range(n_heads)}


def test_dominant_head_bias_negligible() -> None:
    rng = np.random.default_rng(0)
    scores = {
        "winner": rng.normal(-2.0, 0.3, 300),  # dominates by >> noise
        "loser_a": rng.normal(0.0, 0.3, 300),
        "loser_b": rng.normal(0.5, 0.3, 300),
    }
    r = winner_curse_audit(scores, seed=0, n_boot=2000)
    assert r.selected_head == "winner"
    assert r.selection_bias < 1e-3
    assert abs(r.corrected_score - r.naive_score) < 1e-3
    assert r.report()["verdict"] == "bias_negligible"


def test_tied_heads_bias_material_and_corrected() -> None:
    scores = _tied_streams(n_heads=8, n_obs=300, seed=1)
    r = winner_curse_audit(scores, seed=0, n_boot=2000)
    assert r.selection_bias > 0.0
    assert r.corrected_score > r.naive_score
    assert r.report()["verdict"] == "bias_material"
    # honest split-half control: the corrected estimate should sit closer
    # to it than the naive one does
    assert abs(r.corrected_score - r.honest_score) <= abs(r.naive_score - r.honest_score) + 1e-9


def test_determinism() -> None:
    scores = _tied_streams(5, 150, 7)
    a = winner_curse_audit(scores, seed=3, n_boot=500)
    b = winner_curse_audit(scores, seed=3, n_boot=500)
    assert a == b


def test_ci_brackets_point() -> None:
    scores = _tied_streams(4, 200, 2)
    r = winner_curse_audit(scores, seed=0, n_boot=2000)
    lo, hi = r.selection_aware_ci
    assert lo <= r.corrected_score <= hi
    nlo, nhi = r.naive_ci
    assert nlo <= r.naive_score <= nhi


@pytest.mark.parametrize(
    "scores,msg",
    [
        ({}, "at least one head"),
        ({"a": np.array([])}, "empty or non-finite"),
        ({"a": np.array([1.0, np.nan])}, "empty or non-finite"),
        ({"a": np.ones(10), "b": np.ones(9)}, "same number"),
    ],
)
def test_fails_closed(scores: dict[str, np.ndarray], msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        winner_curse_audit(scores)


def test_mc_corrected_closer_to_true_min() -> None:
    """Over 30 seeds of 8 truly-tied heads, corrected beats naive in MAE."""
    errs_naive: list[float] = []
    errs_corr: list[float] = []
    for s in range(30):
        r = winner_curse_audit(_tied_streams(8, 300, 100 + s), seed=200 + s, n_boot=1500)
        errs_naive.append(abs(r.naive_score - 0.0))
        errs_corr.append(abs(r.corrected_score - 0.0))
    assert float(np.mean(errs_corr)) < float(np.mean(errs_naive))


def test_audit_from_streams_report_shape() -> None:
    rng = np.random.default_rng(0)
    rep = audit_from_streams(
        {"a": list(rng.normal(0, 1, 200)), "b": list(rng.normal(0, 1, 200))},
        seed=0,
        n_boot=500,
    )
    assert rep["kind"] == "winner_curse.v1"
    assert rep["research_only"] is True
    assert rep["live_pnl_claim"] is False
    assert len(str(rep["inputs_sha256"])) == 64
    for key in ("naive_score", "corrected_score", "honest_score", "selection_bias"):
        assert np.isfinite(float(rep[key]))
