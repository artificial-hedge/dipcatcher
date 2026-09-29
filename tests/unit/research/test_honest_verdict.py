"""honest_verdict: composite verdict over the sequential-inference lanes."""

from __future__ import annotations

import json

import numpy as np
import pytest

from quant_fund.research.honest_verdict import (
    HONEST_VERDICT_SCHEMA,
    honest_verdict,
    honest_verdict_json,
)


def _streams(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "winner": rng.normal(-0.5, 1.0, 200),
        "runner": rng.normal(0.0, 1.0, 200),
        "far": rng.normal(0.8, 1.0, 200),
    }


def test_report_shape_and_stamps() -> None:
    rep = honest_verdict(_streams(), seed=0, n_boot=300)
    assert rep["kind"] == HONEST_VERDICT_SCHEMA
    assert rep["research_only"] is True
    assert rep["live_pnl_claim"] is False
    # unlabeled streams stamp UNKNOWN, never claim synthetic provenance
    assert rep["data_label"] == "UNKNOWN"
    assert rep["winner"] == "winner"
    assert len(str(rep["inputs_sha256"])) == 64
    assert set(rep["components"]) == {
        "winner_curse",
        "promotion",
        "drift",
        "magnitude",
        "calibration",
        "localize",
    }
    assert rep["verdict"] in {
        "confirmed",
        "supported_with_caveats",
        "not_supported",
        "inconclusive",
    }


def test_verdict_inconclusive_when_lanes_missing_or_confirmed() -> None:
    """On main (lanes on branches) components are unavailable → inconclusive;
    once merged, a dominant winner with stable edge → confirmed-ish."""
    rep = honest_verdict(_streams(), seed=1, n_boot=300)
    unavailable = rep["unavailable_lanes"]
    if unavailable:
        assert rep["verdict"] == "inconclusive"
        assert {"winner_curse", "promotion", "drift"} <= set(unavailable)
    else:
        assert rep["verdict"] in {"confirmed", "supported_with_caveats"}


def test_fallback_winner_is_argmin_mean() -> None:
    scores = _streams(5)
    rep = honest_verdict(scores, seed=0, n_boot=200)
    assert rep["winner"] == "winner"  # lowest mean loss regardless of lanes


def test_determinism() -> None:
    s = _streams(3)
    assert honest_verdict_json(s, seed=2, n_boot=400) == honest_verdict_json(
        _streams(3), seed=2, n_boot=400
    )


def test_json_serializable() -> None:
    payload = json.loads(honest_verdict_json(_streams(7), n_boot=200))
    assert payload["kind"] == HONEST_VERDICT_SCHEMA
    assert all(
        isinstance(payload["components"][k], dict)
        for k in ("winner_curse", "promotion", "drift", "magnitude", "calibration", "localize")
    )


@pytest.mark.parametrize(
    "scores",
    [
        {},
        {"a": np.array([])},
        {"a": np.array([1.0, np.nan])},
        {"a": np.ones(10), "b": np.ones(9)},
        {"a": np.ones(10), "b": np.full(10, np.inf)},
    ],
)
def test_fails_closed(scores: dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError):
        honest_verdict(scores, n_boot=200)


def test_alpha_validated() -> None:
    with pytest.raises(ValueError):
        honest_verdict(_streams(), alpha=1.5, n_boot=200)


def test_extension_lanes_never_veto_core_verdict() -> None:
    """A missing extension lane is recorded but must not flip inconclusive."""
    rep = honest_verdict(_streams(11), seed=0, n_boot=300)
    core_missing = {"winner_curse", "promotion", "drift"} & set(rep["unavailable_lanes"])
    if not core_missing:
        assert rep["verdict"] != "inconclusive"
    # extension lanes recorded when absent
    for lane in ("magnitude", "calibration", "localize"):
        assert lane in rep["components"]


def test_pits_unlock_calibration_lane() -> None:
    scores = _streams(13)
    rng = np.random.default_rng(99)
    rep = honest_verdict(
        scores, pits={h: rng.uniform(0, 1, 200) for h in scores}, seed=0, n_boot=300
    )
    calib = rep["components"]["calibration"]
    if "calibration" in rep["unavailable_lanes"]:
        assert calib == {}  # lane absent on this checkout — honest empty detail
    elif "skipped" not in calib:
        assert "final_evalue" in calib and "miscalibrated" in calib
