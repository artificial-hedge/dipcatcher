"""Keyed per-security volatility: forecasts, units, PIT and the scope contract.

EVERY dataset here is seeded SYNTHETIC data generated in-test and labeled
``source_label="SYNTHETIC"``.  These are correctness tests only — never market
evidence, never a live-trading or profitability claim.

The two proof tests at the bottom are the anti-leakage proofs and are REAL
assertions:
1. no-forward-label proof — shifting the forward labels in time leaves every
   forecast bit-identical;
2. PIT proof — perturbing returns at or after an origin leaves that origin's
   forecast bit-identical (trailing/as-of windows only).

Each proof has a companion MUTANT-harness test that feeds a deliberately
leaky/lookahead implementation to the same proof helper and asserts the helper
raises — so CI keeps proving the proofs can fail on the bugs they target.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from quant_fund.metrics.vol_eval import (
    VARIANCE_UNITS,
    VOLATILITY_UNITS,
    VolUnitsError,
    coerce_units,
    pinball_keyed,
    qlike_keyed,
)
from quant_fund.models.vol_per_security import (
    KEYED_VOL_SPECS,
    KeyedReturnPanel,
    KeyedWalkForwardResult,
    PerSecurityVol,
    walk_forward_per_security,
)
from quant_fund.models.vol_scope import (
    DATE_LEVEL_PORTFOLIO_CONSUMER,
    PER_SECURITY_CONSUMER,
    PER_SECURITY_SCOPE,
    ScopeMismatchError,
)
from quant_fund.models.volatility import GARCHVol

SYNTHETIC = "SYNTHETIC"
KEYS = ("S00", "S01", "S02")
LEVELS = (0.1, 0.5, 0.9)
N_OBS = 240


def _synthetic_keyed_panel(
    seed: int = 20261007,
    keys: tuple[str, ...] = KEYS,
    n_obs: int = N_OBS,
    key_lengths: dict[str, int] | None = None,
) -> KeyedReturnPanel:
    """Seeded SYNTHETIC keyed return panel with forward realized-var labels.

    Label convention (PIT): the label at ``t`` is the realized one-step
    variance ``r[t] ** 2`` — exactly what a forecast issued at ``t`` (using
    returns strictly before ``t``) predicts.
    """
    rng = np.random.default_rng(seed)
    key_rows: list[str] = []
    time_rows: list[int] = []
    return_rows: list[float] = []
    label_rows: list[float] = []
    lengths = dict(key_lengths or {})
    for key in keys:
        size = lengths.get(key, n_obs)
        shock = rng.normal(0.0, 1.0, size)
        scale = 0.008 + 0.004 * float(rng.uniform())
        returns = np.empty(size)
        variance = scale * scale
        for index in range(size):
            if index:
                variance = 2e-7 + 0.08 * returns[index - 1] ** 2 + 0.9 * variance
            returns[index] = np.sqrt(variance) * shock[index]
        labels = returns**2  # forward one-step realized variance at each origin
        key_rows.extend([key] * size)
        time_rows.extend(range(size))
        return_rows.extend(returns.tolist())
        label_rows.extend(labels.tolist())
    return KeyedReturnPanel(
        keys=np.asarray(key_rows),
        times=np.asarray(time_rows, dtype=np.int64),
        returns=np.asarray(return_rows, dtype=float),
        labels=np.asarray(label_rows, dtype=float),
        source_label=SYNTHETIC,
    )


def _fast_walk_forward(panel: KeyedReturnPanel, spec: str = "ewma") -> KeyedWalkForwardResult:
    model = PerSecurityVol(spec, min_obs=20)
    return walk_forward_per_security(
        panel, model, origins=tuple(range(150, 221, 5)), quantiles=LEVELS
    )


def _shift_labels(panel: KeyedReturnPanel, steps: int) -> KeyedReturnPanel:
    """Move every forward label ``steps`` slots later in time, per key."""
    keys = np.asarray(panel.keys)
    labels = np.array(panel.labels, dtype=float)
    for key in panel.key_names():
        rows = np.flatnonzero(keys == key)
        labels[rows[steps:]] = labels[rows[:-steps]]
        labels[rows[:steps]] = np.nan
    return panel.with_labels(labels)


def _perturb_future_returns(panel: KeyedReturnPanel, t0: int, seed: int = 7) -> KeyedReturnPanel:
    """Replace returns at times >= t0 with adversarial junk (per key)."""
    rng = np.random.default_rng(seed)
    times = np.asarray(panel.times)
    returns = np.array(panel.returns, dtype=float)
    returns[times >= t0] = 0.3 + 0.1 * rng.normal(size=int((times >= t0).sum()))
    return panel.with_returns(returns)


def _assert_label_invariance(
    make_forecast: Callable[[KeyedReturnPanel], KeyedWalkForwardResult],
    panel: KeyedReturnPanel,
    steps: int = 3,
) -> None:
    """PROOF HELPER: forecasts must be bit-identical after shifting labels.

    Raises AssertionError on ANY bit difference — this is the no-forward-label
    proof used by the real test and by the mutant-harness test.
    """
    base = make_forecast(panel)
    shifted = make_forecast(_shift_labels(panel, steps))
    assert not np.array_equal(
        np.asarray(panel.labels, dtype=float),
        np.asarray(_shift_labels(panel, steps).labels),
        equal_nan=True,
    ), "the label shift must actually change the labels"
    for name in ("one_step_variance", "cumulative_variance", "quantiles"):
        left = getattr(base, name)
        right = getattr(shifted, name)
        assert (left is None) == (right is None), f"{name} presence changed under label shift"
        if left is not None and right is not None:
            assert np.asarray(left).tobytes() == np.asarray(right).tobytes(), (
                f"{name} changed when labels were shifted forward"
            )


def _assert_future_invariance(
    make_forecast: Callable[[KeyedReturnPanel], KeyedWalkForwardResult],
    panel: KeyedReturnPanel,
    t0: int,
) -> None:
    """PROOF HELPER: forecasts at origins < t0 must ignore returns at t >= t0.

    Raises AssertionError on ANY bit difference — this is the PIT
    trailing-window proof used by the real test and the mutant-harness test.
    """
    base = make_forecast(panel)
    perturbed = make_forecast(_perturb_future_returns(panel, t0))
    keep = np.asarray(base.origins) < t0
    assert keep.any()
    for name in ("one_step_variance", "cumulative_variance", "quantiles"):
        left = np.asarray(getattr(base, name))
        right = np.asarray(getattr(perturbed, name))
        picked_left = left[:, keep] if left.ndim == 2 else left[:, keep, :]
        picked_right = right[:, keep] if right.ndim == 2 else right[:, keep, :]
        assert picked_left.tobytes() == picked_right.tobytes(), (
            f"{name} changed when future returns were perturbed (lookahead)"
        )


def _leaky_walk_forward(panel: KeyedReturnPanel) -> KeyedWalkForwardResult:
    """DELIBERATE LOOKAHEAD MUTANT: forecasts that read the forward labels."""
    result = _fast_walk_forward(panel)
    leak = np.abs(np.nan_to_num(result.realized_variance))
    contaminated = result.cumulative_variance + 1e-3 * leak
    return KeyedWalkForwardResult(
        keys=result.keys,
        origins=result.origins,
        one_step_variance=result.one_step_variance,
        cumulative_variance=contaminated,
        realized_variance=result.realized_variance,
        realized_return=result.realized_return,
        status=result.status,
        horizon=result.horizon,
        quantiles=result.quantiles,
    )


# --- construction and units ------------------------------------------------


def test_panel_records_synthetic_source_label_and_key_axis() -> None:
    panel = _synthetic_keyed_panel()
    assert panel.source_label == SYNTHETIC
    assert panel.key_names() == KEYS
    assert panel.label_at("S00", 10) == pytest.approx(panel.realized_return_at("S00", 10) ** 2)


def test_per_key_forecast_dict_shape_and_units() -> None:
    panel = _synthetic_keyed_panel()
    model = PerSecurityVol("ewma", min_obs=20).fit(panel)
    forecast = model.forecast("S00", horizon=3, quantiles=LEVELS)

    assert forecast["variance"].shape == (3,)
    assert forecast["quantiles"].shape == (3, 3)
    assert forecast["scope"] == PER_SECURITY_SCOPE
    assert forecast["series_scope"] == PER_SECURITY_SCOPE
    assert forecast["variance_units"] == VARIANCE_UNITS == "decimal_squared"
    assert forecast["sigma_units"] == VOLATILITY_UNITS
    assert np.all(forecast["variance"] > 0.0)
    assert np.allclose(forecast["sigma"] ** 2, forecast["variance"])
    assert np.allclose(forecast["cumulative_variance"], np.cumsum(forecast["variance"]))


def test_forecast_variance_is_variance_not_volatility_under_rescaling() -> None:
    """Variance in == variance out: x10 returns -> x100 variance (never x10)."""
    panel = _synthetic_keyed_panel()
    model = PerSecurityVol("ewma", min_obs=20).fit(panel)
    base = model.forecast("S00")["variance"]
    scaled_panel = panel.with_returns(np.asarray(panel.returns, dtype=float) * 10.0)
    scaled = PerSecurityVol("ewma", min_obs=20).fit(scaled_panel).forecast("S00")["variance"]

    assert float(scaled[0]) == pytest.approx(100.0 * float(base[0]))


def test_forecast_units_are_preserved_bit_identically() -> None:
    panel = _synthetic_keyed_panel()
    model = PerSecurityVol("ewma", min_obs=20).fit(panel)
    variance = model.forecast("S00")["variance"]

    coerced = coerce_units(variance, from_units=VARIANCE_UNITS, to_units=VARIANCE_UNITS)
    assert coerced.tobytes() == variance.tobytes()
    with pytest.raises(VolUnitsError):
        coerce_units(variance, from_units=VARIANCE_UNITS, to_units=VOLATILITY_UNITS)


def test_batch_view_preserves_per_key_variance_bits() -> None:
    """KeyedVarianceForecast.forecast(key) exposes the batch variance unchanged."""
    panel = _synthetic_keyed_panel()
    batch = PerSecurityVol("ewma", min_obs=20).fit(panel).forecast_keys(horizon=2)
    row = batch.variance[batch.keys.index("S01")]
    view = batch.forecast("S01")

    assert view["variance"].tobytes() == row.tobytes()
    assert view["variance_units"] == VARIANCE_UNITS


def test_ewma_forecast_matches_the_documented_recursion() -> None:
    panel = _synthetic_keyed_panel()
    model = PerSecurityVol("ewma", lam=0.94, min_obs=20).fit(panel)
    history = panel.key_returns("S01")
    forecast = model.forecast("S01")["variance"]

    weights = np.empty(history.size)
    weights[0] = history[0] ** 2
    for index in range(1, history.size):
        weights[index] = 0.94 * weights[index - 1] + 0.06 * history[index - 1] ** 2
    expected = 0.94 * weights[-1] + 0.06 * history[-1] ** 2

    assert float(forecast[0]) == pytest.approx(float(expected), rel=1e-12)


def test_har_and_garch_specs_produce_finite_variance_forecasts() -> None:
    panel = _synthetic_keyed_panel()
    for spec in ("har", "garch"):
        model = PerSecurityVol(spec, min_obs=30).fit(panel)
        forecast = model.forecast("S02", horizon=2)
        assert forecast["fit_status"] == "ok"
        assert np.all(np.isfinite(forecast["variance"]))
        assert np.all(forecast["variance"] > 0.0)


def test_qmle_specs_are_one_step_only_and_label_free() -> None:
    panel = _synthetic_keyed_panel()
    for spec in ("aparch", "figarch"):
        assert spec in KEYED_VOL_SPECS
        model = PerSecurityVol(spec, min_obs=50).fit(panel)
        forecast = model.forecast("S00", horizon=1)
        assert forecast["fit_status"] == "ok"
        assert np.isfinite(forecast["variance"][0])
        with pytest.raises(ValueError, match="horizon=1 only"):
            model.forecast("S00", horizon=2)


# --- honest-NaN per-key isolation ------------------------------------------


def test_one_bad_key_is_nan_without_poisoning_the_other_keys() -> None:
    merged = _synthetic_keyed_panel(
        keys=("GOOD1", "BAD", "GOOD2"), key_lengths={"GOOD1": 120, "BAD": 5, "GOOD2": 120}
    )
    model = PerSecurityVol("ewma", min_obs=20).fit(merged)
    batch = model.forecast_keys(horizon=2)

    rows = {key: index for index, key in enumerate(batch.keys)}
    assert np.isnan(batch.variance[rows["BAD"]]).all()
    assert batch.status[rows["BAD"]].startswith("failed:insufficient_observations")
    for key in ("GOOD1", "GOOD2"):
        assert np.all(np.isfinite(batch.variance[rows[key]]))
        assert batch.status[rows[key]] == "ok"
    assert np.isnan(batch.means[rows["BAD"]])
    assert np.isfinite(batch.means[rows["GOOD1"]])


def test_non_finite_key_fails_honestly_alone() -> None:
    panel = _synthetic_keyed_panel(keys=("OKA", "OKB"), n_obs=120)
    broken = np.asarray(panel.returns, dtype=float).copy()
    broken[np.asarray(panel.keys) == "OKB"] = np.nan
    model = PerSecurityVol("ewma", min_obs=20).fit(panel.with_returns(broken))
    batch = model.forecast_keys()

    rows = {key: index for index, key in enumerate(batch.keys)}
    assert np.isnan(batch.variance[rows["OKB"]]).all()
    assert np.isfinite(batch.variance[rows["OKA"]]).all()


def test_unknown_key_and_bad_requests_fail_closed() -> None:
    panel = _synthetic_keyed_panel()
    model = PerSecurityVol("ewma", min_obs=20).fit(panel)
    with pytest.raises(ValueError, match="not fitted"):
        model.forecast("ZZZ")
    with pytest.raises(ValueError, match="horizon"):
        model.forecast("S00", horizon=0)
    with pytest.raises(ValueError, match="quantiles"):
        model.forecast("S00", quantiles=(0.0, 0.5))
    with pytest.raises(ValueError, match="spec"):
        PerSecurityVol("nope")
    with pytest.raises(ValueError, match="origins"):
        walk_forward_per_security(panel, model, origins=())


# --- scope contract on the keyed lane --------------------------------------


def test_per_security_artifact_admits_only_per_security_consumers() -> None:
    model = PerSecurityVol("ewma")
    assert model.series_scope == PER_SECURITY_SCOPE
    model.assert_consumer_scope(PER_SECURITY_CONSUMER)
    with pytest.raises(ScopeMismatchError):
        model.assert_consumer_scope(DATE_LEVEL_PORTFOLIO_CONSUMER)


def test_pooled_artifact_rejects_per_security_consumers_symmetrically() -> None:
    pooled = GARCHVol(series_scope="date_level_equal_weight_cross_section")
    with pytest.raises(ScopeMismatchError):
        pooled.assert_consumer_scope(PER_SECURITY_CONSUMER)


# --- walk-forward, scoring and reproducibility -----------------------------


def test_walk_forward_is_scored_with_qlike_and_pinball_on_synthetic_keys() -> None:
    panel = _synthetic_keyed_panel()
    result = _fast_walk_forward(panel)
    assert result.source_label == SYNTHETIC

    qlike = qlike_keyed(result.realized_variance, result.cumulative_variance, keys=result.keys)
    assert np.isfinite(qlike["qlike_mean"])
    pinball = pinball_keyed(result.realized_return, result.quantiles, LEVELS, keys=result.keys)
    assert all(np.isfinite(value) for value in pinball["pinball_mean_per_tau"].values())
    assert np.isfinite(pinball["pinball_per_key"]["S00"]["0.5"])


def test_walk_forward_is_deterministic_across_runs() -> None:
    panel = _synthetic_keyed_panel()
    first = _fast_walk_forward(panel, spec="garch")
    second = _fast_walk_forward(panel, spec="garch")
    assert first.cumulative_variance.tobytes() == second.cumulative_variance.tobytes()
    assert first.one_step_variance.tobytes() == second.one_step_variance.tobytes()


def test_walk_forward_statuses_are_recorded_per_key_and_origin() -> None:
    panel = _synthetic_keyed_panel()
    result = _fast_walk_forward(panel)
    assert set(result.status) == set(KEYS)
    assert all(rows == ("ok",) * result.origins.size for rows in result.status.values())


# --- the anti-leakage proofs (real assertions) + mutant harnesses ----------


def test_no_forward_label_proof_shifting_labels_is_bit_identical() -> None:
    """PROOF: shifting forward labels leaves every forecast bit-identical."""
    panel = _synthetic_keyed_panel()
    _assert_label_invariance(_fast_walk_forward, panel)


def test_no_forward_label_proof_harness_detects_a_leaky_mutant() -> None:
    """MUTANT HARNESS: the proof above MUST fail on a label-reading bug.

    ``_leaky_walk_forward`` deliberately contaminates forecasts with the
    forward labels (a lookahead bug). Feeding it to the SAME proof helper
    raises AssertionError — which proves the no-forward-label proof can fail
    and is therefore a real proof, not a comment.
    """
    panel = _synthetic_keyed_panel()
    with pytest.raises(AssertionError, match="labels"):
        _assert_label_invariance(_leaky_walk_forward, panel)


def test_pit_proof_future_returns_cannot_change_past_forecasts() -> None:
    """PROOF: returns at t >= t0 cannot change forecasts issued before t0."""
    panel = _synthetic_keyed_panel()
    _assert_future_invariance(_fast_walk_forward, panel, t0=210)


def test_pit_proof_harness_detects_a_lookahead_mutant() -> None:
    """MUTANT HARNESS: the PIT proof MUST fail on an as-of-ignoring bug.

    ``_lookahead_walk_forward`` deliberately refits on the FULL return panel
    (ignoring the as-of boundary). The SAME proof helper raises AssertionError
    on it — proving the PIT proof can fail on the lookahead bug it targets.
    """
    panel = _synthetic_keyed_panel()
    with pytest.raises(AssertionError, match="lookahead"):
        _assert_future_invariance(_lookahead_walk_forward, panel, t0=210)


def _lookahead_walk_forward(panel: KeyedReturnPanel) -> KeyedWalkForwardResult:
    """DELIBERATE LOOKAHEAD MUTANT: refits on the full panel, ignoring as-of."""
    origins = tuple(range(150, 221, 5))
    model = PerSecurityVol("ewma", min_obs=20)
    fitted = model.fit(panel)  # no as_of -> full history (the bug)
    batch = fitted.forecast_keys(horizon=1, quantiles=LEVELS)
    n_origins = len(origins)
    return KeyedWalkForwardResult(
        keys=batch.keys,
        origins=np.asarray(origins, dtype=np.int64),
        one_step_variance=np.repeat(batch.variance, n_origins, axis=1),
        cumulative_variance=np.repeat(batch.variance, n_origins, axis=1),
        realized_variance=np.full((len(batch.keys), n_origins), np.nan),
        realized_return=np.full((len(batch.keys), n_origins), np.nan),
        status={key: tuple(["ok"] * n_origins) for key in batch.keys},
        horizon=1,
        quantiles=np.repeat(batch.quantiles, n_origins, axis=1),
    )


# --- panel validation ------------------------------------------------------


def test_panel_rejects_malformed_inputs() -> None:
    keys = np.asarray(["A", "A"])
    times = np.asarray([0, 1], dtype=np.int64)
    returns = np.asarray([0.01, 0.02])
    with pytest.raises(ValueError, match="same length"):
        KeyedReturnPanel(keys=keys, times=times, returns=returns[:1])
    with pytest.raises(ValueError, match="non-empty strings"):
        KeyedReturnPanel(keys=np.asarray(["A", ""]), times=times, returns=returns)
    with pytest.raises(ValueError, match="strictly increasing"):
        KeyedReturnPanel(keys=keys, times=np.asarray([1, 1], dtype=np.int64), returns=returns)
    with pytest.raises(ValueError, match="labels"):
        KeyedReturnPanel(keys=keys, times=times, returns=returns, labels=np.zeros(5))


def test_trailing_uses_strictly_prior_returns_only() -> None:
    panel = _synthetic_keyed_panel(keys=("ONLY",), n_obs=30)
    trailing = panel.trailing("ONLY", as_of=10)
    assert trailing.size == 10
    assert np.allclose(trailing, panel.key_returns("ONLY")[:10])


def test_diagnostics_declare_scope_units_and_label_free_fit() -> None:
    panel = _synthetic_keyed_panel()
    model = PerSecurityVol("ewma", min_obs=20).fit(panel)
    diagnostics: dict[str, Any] = model.diagnostics()
    assert diagnostics["series_scope"] == PER_SECURITY_SCOPE
    assert diagnostics["variance_units"] == VARIANCE_UNITS
    assert diagnostics["label_usage"] == "none_forecasts_are_label_free"
    assert diagnostics["pit_contract"] == "trailing_returns_strictly_before_as_of"
    assert set(diagnostics["status"]) == set(KEYS)
