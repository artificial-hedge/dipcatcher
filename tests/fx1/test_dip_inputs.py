"""Synthetic regressions for fail-closed Dip Quality bench inputs."""

from __future__ import annotations

import math
from typing import Any

import pytest

from fx1.bench.dip import (
    DipEvent,
    DipForecast,
    assert_bench_output_honest,
    detect_dip_events,
    evaluate_forecasts,
    unconditional_baseline,
)
from fx1.bench.dip_audit import dip_audit

pytestmark = pytest.mark.synthetic


@pytest.mark.parametrize("price", [0.0, -0.0, -1.0])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_detection_rejects_nonpositive_closes(price: float, index: int) -> None:
    closes = [100.0, 80.0, 100.0]
    closes[index] = price
    with pytest.raises(ValueError, match=f"close at index {index}.*strictly positive"):
        detect_dip_events(closes, ["a", "b", "c"], "T", horizons_bars={"h": 1})


def test_zero_peak_fails_closed_before_drawdown_division() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        detect_dip_events([0.0, -1.0], ["a", "b"], "T")


@pytest.mark.parametrize("bars", [0, -1, True, False, 1.0, 1.5, math.nan, math.inf, "1"])
@pytest.mark.parametrize("closes", [[], [100.0, 101.0], [100.0, 80.0, 100.0]])
def test_horizons_require_positive_integers_even_without_events(
    bars: Any, closes: list[float]
) -> None:
    with pytest.raises(ValueError, match="horizon 'h' bars must be a positive integer"):
        detect_dip_events(
            closes, [str(i) for i in range(len(closes))], "T", horizons_bars={"h": bars}
        )


def test_recovery_includes_final_bar_and_preserves_censoring() -> None:
    events = detect_dip_events(
        [100.0, 80.0, 90.0, 100.0],
        ["a", "b", "c", "d"],
        "T",
        horizons_bars={"one": 1, "two": 2, "three": 3},
    )
    assert len(events) == 1
    assert events[0].recovered == {"one": False, "two": True, "three": None}


@pytest.mark.parametrize("n_bins", [0, -1, True, False, 1.0, 1.5, math.nan, math.inf, "2"])
@pytest.mark.parametrize("outcome", ["empty", "censored", "observable"])
def test_calibration_bins_require_positive_integers(n_bins: Any, outcome: str) -> None:
    events = (
        []
        if outcome == "empty"
        else [DipEvent("T", "p", "t", 0.2, {"h": None if outcome == "censored" else True})]
    )
    forecasts = [] if outcome == "empty" else [DipForecast("T", "t", {"h": 0.5})]
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        evaluate_forecasts(events, forecasts, n_bins=n_bins)


@pytest.mark.parametrize("second_flag", [True, False, None])
def test_duplicate_events_fail_closed_in_scores_and_baseline(second_flag: bool | None) -> None:
    events = [
        DipEvent("T", "p", "t", 0.2, {"h": True}),
        DipEvent("T", "other-peak", "t", 0.3, {"h": second_flag}),
    ]
    with pytest.raises(ValueError, match="duplicate event identity"):
        unconditional_baseline(events)
    with pytest.raises(ValueError, match="duplicate event identity"):
        evaluate_forecasts(events, [DipForecast("T", "t", {"h": 1.0})])
    with pytest.raises(ValueError, match="duplicate event identity"):
        evaluate_forecasts(events, [])


@pytest.mark.parametrize("probabilities", [{"h": 0.5}, {"h": 0.9}, {"later": 0.2}, {}])
def test_duplicate_forecasts_cannot_weight_or_split_an_event(
    probabilities: dict[str, float],
) -> None:
    events = [DipEvent("T", "p", "t", 0.2, {"h": True, "later": None})]
    forecasts = [DipForecast("T", "t", {"h": 0.5}), DipForecast("T", "t", probabilities)]
    with pytest.raises(ValueError, match="duplicate forecast identity"):
        evaluate_forecasts(events, forecasts)


@pytest.mark.parametrize("probability", [-0.1, 1.1, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("flag", [True, None])
def test_invalid_probabilities_fail_even_for_censored_horizons(
    probability: float, flag: bool | None
) -> None:
    with pytest.raises(ValueError, match="probability out of range"):
        evaluate_forecasts(
            [DipEvent("T", "p", "t", 0.2, {"h": flag})],
            [DipForecast("T", "t", {"h": probability})],
        )


def test_distinct_assets_and_dates_preserve_proper_scores() -> None:
    events = [
        DipEvent("A", "p", "t", 0.2, {"h": True, "later": None}),
        DipEvent("B", "p", "t", 0.2, {"h": False, "later": None}),
        DipEvent("A", "p", "t2", 0.2, {"h": False, "later": None}),
    ]
    forecasts = [
        DipForecast("A", "t", {"h": 0.75, "later": 0.9}),
        DipForecast("B", "t", {"h": 0.25, "later": 0.9}),
        DipForecast("A", "t2", {"h": 0.25, "later": 0.9}),
    ]
    metrics = evaluate_forecasts(events, forecasts, n_bins=4)
    assert metrics == pytest.approx(
        {
            "brier_h": 0.0625,
            "log_loss_h": -math.log(0.75),
            "ece_h": 0.25,
            "n_h": 3.0,
            "brier_overall": 0.0625,
        }
    )
    assert unconditional_baseline(events) == {"h": 1 / 3}
    assert_bench_output_honest(metrics)


def test_valid_empty_scores_stay_empty() -> None:
    assert evaluate_forecasts([], [], n_bins=1) == {}
    assert unconditional_baseline([]) == {}


def test_dip_audit_matches_current_boundary_and_unknown_event_contract() -> None:
    report = dip_audit()
    assert report["detection"]["boundary_out"] == {"h1": False, "h2": True, "h3": None}
    assert report["detection"]["n_events"] == 1
    assert report["detection"]["rearm"] is True
    assert report["scoring"]["ghost_rejected"] == "raise:ValueError"
    assert report["scoring"]["baseline_h1"] == 0.5
    assert report["scoring"]["baseline_h2"] == 1.0
    assert report["scoring"]["h2_skips_unobservable"] is True
    assert report["scoring"]["oor_prob_raises"] == "raise:ValueError"
    assert report["scoring"]["brier_overall_absent_when_empty"] is True
    outcomes = {case["name"]: case["outcome"] for case in report["honesty"]["cases"]}
    assert outcomes["pl_embedded"] == "raise:ValueError"
    assert outcomes["camel_ratio"] == "raise:ValueError"
    assert outcomes["clean"] == "accepted"
