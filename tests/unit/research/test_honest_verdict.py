"""honest_verdict: composite verdict over the sequential-inference lanes."""

from __future__ import annotations

import json
import math

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
    """Any unavailable lane forces inconclusive; once every lane merges,
    a dominant winner with stable edge → confirmed-ish."""
    rep = honest_verdict(_streams(), seed=1, n_boot=300)
    unavailable = rep["unavailable_lanes"]
    if unavailable:
        assert rep["verdict"] == "inconclusive"
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


def test_any_unavailable_lane_forces_inconclusive() -> None:
    """A missing lane — core or extension — is recorded AND vetoes: an
    un-runnable component cannot vouch for the claim."""
    rep = honest_verdict(_streams(11), seed=0, n_boot=300)
    if rep["unavailable_lanes"]:
        assert rep["verdict"] == "inconclusive"
    # extension lanes recorded when absent
    for lane in ("magnitude", "calibration", "localize"):
        assert lane in rep["components"]


def test_unavailable_lanes_recorded_and_nonempty_means_inconclusive() -> None:
    """Skipped lanes (no runner-up, pits not supplied, no drift) do NOT
    veto — only a lane that cannot be imported/executed does."""
    rep = honest_verdict(_streams(11), seed=0, n_boot=300)
    # on a checkout without every extension module this must be inconclusive
    assert (rep["verdict"] == "inconclusive") == bool(rep["unavailable_lanes"])
    calib = rep["components"]["calibration"]
    if calib.get("skipped"):
        assert "calibration" not in rep["unavailable_lanes"]


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


def test_dataset_sha256_tracks_stream_content() -> None:
    """Same loss streams share dataset_sha256 regardless of seed/n_boot;
    a mutated stream changes it."""
    r1 = honest_verdict(_streams(3), seed=0, n_boot=100)
    r2 = honest_verdict(_streams(3), seed=9, n_boot=150)
    d1, d2 = r1["dataset_sha256"], r2["dataset_sha256"]
    assert len(d1) == 64 and all(c in "0123456789abcdef" for c in d1)
    assert d1 == d2
    alt = _streams(3)
    alt["winner"] = np.asarray(alt["winner"], dtype=float) * 2.0
    r3 = honest_verdict(alt, seed=0, n_boot=100)
    assert r3["dataset_sha256"] != d1


def test_dataset_sha256_includes_supplied_pits() -> None:
    def _pits(seed: int, heads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        return {h: rng.uniform(0, 1, 200) for h in heads}

    scores = _streams(4)
    without = honest_verdict(scores, seed=0, n_boot=100)
    with_pits = honest_verdict(scores, pits=_pits(17, scores), seed=0, n_boot=100)
    same_pits = honest_verdict(scores, pits=_pits(17, scores), seed=5, n_boot=150)
    assert with_pits["dataset_sha256"] != without["dataset_sha256"]
    assert with_pits["dataset_sha256"] == same_pits["dataset_sha256"]


def test_page_hinkley_only_alarm_demotes_confirmed() -> None:
    """Declared gate: a PageHinkley alarm without an e-process crossing is
    the 'drift diagnostic fired while the e-process did not' case —
    supported_with_caveats, not confirmed."""
    rng = np.random.default_rng(0)
    winner = rng.normal(-1.5, 1.0, 200)
    winner[100:] += 1.0  # mid-stream level shift: PH alarms, e-process quiet
    scores = {
        "winner": winner,
        "runner": rng.normal(0.0, 1.0, 200),
        "far": rng.normal(0.8, 1.0, 200),
    }
    rep = honest_verdict(scores, seed=0, n_boot=500)
    drift = rep["components"]["drift"]
    assert drift["page_hinkley_alarmed"] is True
    assert drift["eprocess_alarmed"] is False
    assert rep["components"]["promotion"]["promoted"] is True
    assert rep["verdict"] == "supported_with_caveats"


def test_reported_drift_evalue_capped_finite() -> None:
    """A strongly drifting stream drives log_e past 700; the reported
    final_evalue must stay finite (strict JSON has no Infinity literal)."""
    n = 2500
    scores = {
        "winner": np.linspace(0.0, 1.0, n) ** 3 - 10.0,  # increasing diffs
        "runner": np.random.default_rng(2).normal(0.0, 1.0, n),
    }
    rep = honest_verdict(scores, seed=0, n_boot=200)
    fe = rep["components"]["drift"]["final_evalue"]
    assert math.isfinite(fe)
    json.dumps(rep, default=float, allow_nan=False)
