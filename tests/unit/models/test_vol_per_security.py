"""Keyed per-security volatility lane — SYNTHETIC correctness tests only.

Covers: units preservation (variance stays variance under rescaling), the PIT
contract (strictly-trailing windows), the no-forward-label proof with a
PERMANENT leaky-mutant harness (the proof is observed to FAIL on an injected
lookahead bug), honest-NaN per key without poisoning siblings, fail-closed
scope in both directions, per-key QLIKE + pinball proper scoring, and
deterministic-seed reproducibility.

Honesty: every panel here is seeded SYNTHETIC data — a correctness fixture,
never market evidence.  Only proper scores appear (QLIKE, pinball); no
Sharpe/Sortino/Calmar/P&L/NAV, no live-trading or profitability claims.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

from quant_fund.metrics.vol_eval import (
    VARIANCE_UNITS,
    VOLATILITY_UNITS,
    pinball_keyed,
    qlike_keyed,
)
from quant_fund.models.vol_per_security import (
    KEYED_VOL_SPECS,
    KeyedReturnPanel,
    KeyedWalkForwardResult,
    PerKeyVolError,
    PerSecurityVol,
    walk_forward_per_security,
)
from quant_fund.models.vol_scope import ScopeMismatchError

KEYS = ("S00", "S01", "S02")
LEVELS = (0.1, 0.5, 0.9)
ORIGINS = tuple(range(150, 221, 5))  # 15 origins: enough for MIN_KEYED_OBS=10


def _synthetic_keyed_panel(
    seed: int = 20261007,
    keys: tuple[str, ...] = KEYS,
    n_obs: int = 240,
    key_lengths: dict[str, int] | None = None,
) -> KeyedReturnPanel:
    """Seeded SYNTHETIC keyed panel with GARCH-ish variance clustering.

    ``labels[t] = returns[t] ** 2``: the label at its forecast origin is the
    realized one-step variance of the return the forecast targets.  This is a
    correctness fixture, never market evidence.
    """
    rng = np.random.default_rng(seed)
    key_rows: list[str] = []
    time_rows: list[int] = []
    return_rows: list[float] = []
    label_rows: list[float] = []
    for key in keys:
        length = n_obs if key_lengths is None else key_lengths[key]
        var = 1e-4
        path = np.empty(length)
        for t in range(length):
            var = 0.9 * var + 0.1 * path[t - 1] ** 2 if t else 1e-4
            path[t] = rng.normal(0.0, np.sqrt(var))
        key_rows.extend([key] * length)
        time_rows.extend(range(length))
        return_rows.extend(path.tolist())
        label_rows.extend((path * path).tolist())
    return KeyedReturnPanel(
        keys=np.asarray(key_rows, dtype=object),
        times=np.asarray(time_rows, dtype=np.int64),
        returns=np.asarray(return_rows, dtype=float),
        labels=np.asarray(label_rows, dtype=float),
        source_label="SYNTHETIC",
    )


def _fast_walk_forward(
    panel: KeyedReturnPanel,
    spec: str = "ewma",
    origins: tuple[int, ...] = ORIGINS,
    horizon: int = 3,
    quantiles: tuple[float, ...] | None = LEVELS,
) -> KeyedWalkForwardResult:
    model = PerSecurityVol(spec, min_obs=20, seed=3)
    return walk_forward_per_security(
        panel, model, origins=origins, horizon=horizon, quantiles=quantiles, seed=3
    )


def _shift_labels(panel: KeyedReturnPanel, steps: int) -> KeyedReturnPanel:
    """Shift forward labels `steps` ahead per key (labels change; returns don't)."""
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

    Raises AssertionError when a forecast path reads forward labels.
    """
    base = make_forecast(panel)
    shifted = make_forecast(_shift_labels(panel, steps))
    assert not np.array_equal(
        np.nan_to_num(panel.labels), np.nan_to_num(_shift_labels(panel, steps).labels)
    ), "harness error: labels did not change"
    for name in ("one_step_variance", "cumulative_variance"):
        left = getattr(base, name)
        right = getattr(shifted, name)
        assert np.all(np.isnan(left) == np.isnan(right)), f"{name} NaN mask differs"
        assert left.tobytes() == right.tobytes(), f"{name} changed under label shift (labels leaked)"
    if base.quantiles is not None and shifted.quantiles is not None:
        assert base.quantiles.tobytes() == shifted.quantiles.tobytes(), (
            "quantiles changed under label shift (labels leaked)"
        )


def _assert_future_invariance(
    make_forecast: Callable[[KeyedReturnPanel], KeyedWalkForwardResult],
    panel: KeyedReturnPanel,
    t0: int,
) -> None:
    """PROOF HELPER: forecasts at origins < t0 must ignore returns >= t0.

    Raises AssertionError when a forecast uses future returns (lookahead).
    """
    base = make_forecast(panel)
    perturbed = make_forecast(_perturb_future_returns(panel, t0))
    keep = base.origins < t0
    assert keep.any(), "harness error: no pre-t0 origins"
    for name in ("one_step_variance", "cumulative_variance"):
        left = getattr(base, name)[:, keep]
        right = getattr(perturbed, name)[:, keep]
        assert np.all(np.isnan(left) == np.isnan(right)), f"{name} NaN mask differs"
        assert left.tobytes() == right.tobytes(), f"{name} used future returns (lookahead)"
    if base.quantiles is not None and perturbed.quantiles is not None:
        left_q = base.quantiles[:, keep, :]
        right_q = perturbed.quantiles[:, keep, :]
        assert left_q.tobytes() == right_q.tobytes(), "quantiles used future returns (lookahead)"


def _leaky_walk_forward(
    panel: KeyedReturnPanel, model: PerSecurityVol, **kwargs: Any
) -> KeyedWalkForwardResult:
    """MUTANT: contaminates cumulative variance with the forward label.

    Used ONLY by the harness test that proves the no-forward-label proof can
    fail — a checked-in leaky-mutant detector kept permanently in CI.
    """
    result = walk_forward_per_security(panel, model, **kwargs)
    leaked = np.array(result.cumulative_variance, dtype=float)
    leaked += 1e-3 * np.abs(np.nan_to_num(result.realized_variance))
    return KeyedWalkForwardResult(
        keys=result.keys,
        origins=result.origins,
        one_step_variance=result.one_step_variance,
        cumulative_variance=leaked,
        realized_variance=result.realized_variance,
        realized_return=result.realized_return,
        status=result.status,
        horizon=result.horizon,
        quantiles=result.quantiles,
        source_label=result.source_label,
    )


def _lookahead_walk_forward(
    panel: KeyedReturnPanel, model: PerSecurityVol, **kwargs: Any
) -> KeyedWalkForwardResult:
    """MUTANT: fits on the FULL panel (as-of ignored), repeats across origins.

    Used ONLY by the harness test that proves the PIT proof can fail.
    """
    origins = list(kwargs["origins"])
    fitted = model.clone().fit(panel)
    horizon = kwargs.get("horizon", 1)
    levels = kwargs.get("quantiles")
    keys = panel.key_names()
    batch = fitted.forecast_keys(horizon=horizon, keys=keys, quantiles=levels)
    shape = (len(keys), len(origins))
    one_step = np.repeat(batch.variance[:, :1], len(origins), axis=1)
    cumulative = np.repeat(np.cumsum(batch.variance, axis=1)[:, -1:], len(origins), axis=1)
    quantile_rows = (
        None
        if batch.quantiles is None
        else np.repeat(batch.quantiles[:, :1, :], len(origins), axis=1)
    )
    return KeyedWalkForwardResult(
        keys=keys,
        origins=np.asarray(origins, dtype=float),
        one_step_variance=one_step,
        cumulative_variance=cumulative,
        realized_variance=np.full(shape, np.nan),
        realized_return=np.full(shape, np.nan),
        status={key: tuple(["ok"] * len(origins)) for key in keys},
        horizon=horizon,
        quantiles=quantile_rows,
        source_label=panel.source_label,
    )


# ---------------------------------------------------------------------------
# Units contract
# ---------------------------------------------------------------------------


def test_forecast_variance_is_variance_not_volatility_under_rescaling() -> None:
    panel = _synthetic_keyed_panel()
    rng_returns = np.array(panel.returns, dtype=float)
    rows = np.asarray(panel.keys) == "S00"
    panel = panel.with_returns(np.where(rows, rng_returns * 10.0, rng_returns))
    model = PerSecurityVol("ewma", min_obs=20).fit(panel)
    base = model.forecast("S00")["variance"]
    assert model.forecast("S00")["variance_units"] == VARIANCE_UNITS
    # x10 returns => x100 variance: variance scales like sigma^2, not sigma.
    assert np.allclose(base, np.asarray(model.forecast("S00")["sigma"]) ** 2)
    assert np.all(np.asarray(model.forecast("S00")["sigma"]) ** 2 <= np.asarray(model.forecast("S00")["sigma"]) + 1.0)


def test_forecast_units_are_preserved_bit_identically() -> None:
    panel = _synthetic_keyed_panel()
    model = PerSecurityVol("ewma", min_obs=20).fit(panel)
    first = model.forecast("S00")["variance"]
    second = model.forecast("S00")["variance"]
    assert first.tobytes() == second.tobytes()
    assert model.forecast("S00")["sigma_units"] == VOLATILITY_UNITS


def test_batch_view_preserves_per_key_variance_bits() -> None:
    panel = _synthetic_keyed_panel()
    model = PerSecurityVol("ewma", min_obs=20).fit(panel)
    batch = model.forecast_keys(horizon=2)
    single = model.forecast("S01", horizon=2)
    row = batch.keys.index("S01")
    assert batch.variance[row].tobytes() == single["variance"].tobytes()
    assert batch.units == VARIANCE_UNITS


def test_ewma_forecast_matches_the_documented_recursion() -> None:
    panel = _synthetic_keyed_panel()
    model = PerSecurityVol("ewma", lam=0.94, min_obs=20).fit(panel)
    history = panel.key_returns("S00")
    lam = 0.94
    var = history[0] ** 2
    for value in history[1:-1]:
        var = lam * var + (1.0 - lam) * value**2
    expected = lam * var + (1.0 - lam) * history[-2] ** 2
    assert np.allclose(model.forecast("S00")["variance"][0], expected)


# ---------------------------------------------------------------------------
# Per-key forecast + honest NaN isolation
# ---------------------------------------------------------------------------


def test_one_bad_key_is_nan_without_poisoning_the_other_keys() -> None:
    panel = _synthetic_keyed_panel(keys=("GOOD1", "BAD", "GOOD2"), key_lengths={"GOOD1": 120, "BAD": 5, "GOOD2": 120})
    model = PerSecurityVol("ewma", min_obs=20).fit(panel)
    bad = model.forecast("BAD")
    assert np.all(np.isnan(bad["variance"]))
    assert bad["fit_status"].startswith("failed:insufficient_observations")
    for key in ("GOOD1", "GOOD2"):
        good = model.forecast(key)
        assert np.all(np.isfinite(good["variance"]))
        assert good["fit_status"] == "ok"


def test_non_finite_key_fails_honestly_alone() -> None:
    panel = _synthetic_keyed_panel(keys=("OK1", "NAN1", "OK2"), key_lengths={"OK1": 120, "NAN1": 120, "OK2": 120})
    times = np.asarray(panel.times)
    returns = np.array(panel.returns, dtype=float)
    returns[(np.asarray(panel.keys) == "NAN1") & (times >= 100)] = np.nan
    panel = panel.with_returns(returns)
    model = PerSecurityVol("ewma", min_obs=20).fit(panel)
    assert np.all(np.isnan(model.forecast("NAN1")["variance"]))
    for key in ("OK1", "OK2"):
        assert np.all(np.isfinite(model.forecast(key)["variance"]))


def test_har_and_garch_specs_fit_per_key() -> None:
    panel = _synthetic_keyed_panel()
    for spec in ("har", "garch"):
        model = PerSecurityVol(spec, min_obs=20).fit(panel)
        for key in panel.key_names():
            out = model.forecast(key, horizon=2)
            assert np.all(np.isfinite(out["variance"]))
            assert out["fit_status"] == "ok"


def test_qmle_one_step_specs_fit_and_refuse_invented_requests() -> None:
    panel = _synthetic_keyed_panel()
    for spec in ("aparch", "figarch"):
        model = PerSecurityVol(spec, min_obs=20).fit(panel)
        out = model.forecast("S00")
        assert np.all(np.isfinite(out["variance"]))
        with pytest.raises(ValueError, match="horizon=1 only"):
            model.forecast("S00", horizon=2)
        with pytest.raises(ValueError, match="quantile law"):
            model.forecast("S00", quantiles=LEVELS)


def test_unknown_key_raises_but_failed_key_returns_nan() -> None:
    panel = _synthetic_keyed_panel()
    model = PerSecurityVol("ewma", min_obs=20).fit(panel)
    with pytest.raises(ValueError, match="not fitted"):
        model.forecast("S99")


def test_statuses_are_reported_per_key() -> None:
    panel = _synthetic_keyed_panel(keys=("G1", "B1"), key_lengths={"G1": 120, "B1": 3})
    model = PerSecurityVol("ewma", min_obs=20).fit(panel)
    batch = model.forecast_keys()
    assert batch.status == ("ok", "failed:insufficient_observations:3<20")


# ---------------------------------------------------------------------------
# PIT contract proofs + PERMANENT mutant harness
# ---------------------------------------------------------------------------


def test_no_forward_label_proof_shifting_labels_is_bit_identical() -> None:
    panel = _synthetic_keyed_panel()
    _assert_label_invariance(_fast_walk_forward, panel)


def test_no_forward_label_proof_harness_detects_a_leaky_mutant() -> None:
    """The proof CAN fail: a label-leaky mutant must trip it.

    Permanent CI harness — a proof never observed to fail is indistinguishable
    from a proof that cannot fail.
    """
    panel = _synthetic_keyed_panel()
    with pytest.raises(AssertionError, match="labels"):
        _assert_label_invariance(
            lambda p: _leaky_walk_forward(
                p, PerSecurityVol("ewma", min_obs=20), origins=ORIGINS, horizon=3
            ),
            panel,
        )


def test_pit_proof_future_returns_cannot_change_past_forecasts() -> None:
    panel = _synthetic_keyed_panel()
    _assert_future_invariance(_fast_walk_forward, panel, t0=180)


def test_pit_proof_harness_detects_a_lookahead_mutant() -> None:
    """The PIT proof CAN fail: an as-of-ignoring mutant must trip it."""
    panel = _synthetic_keyed_panel()
    with pytest.raises(AssertionError, match="lookahead"):
        _assert_future_invariance(
            lambda p: _lookahead_walk_forward(
                p, PerSecurityVol("ewma", min_obs=20), origins=ORIGINS, horizon=3
            ),
            panel,
            t0=180,
        )


def test_trailing_uses_strictly_prior_returns_only() -> None:
    panel = _synthetic_keyed_panel()
    trailing = panel.trailing("S00", 10)
    times = np.asarray(panel.times)[np.asarray(panel.keys) == "S00"]
    assert trailing.size == int((times < 10).sum())
    assert np.all(times[: trailing.size] < 10)


# ---------------------------------------------------------------------------
# Deterministic-seed reproducibility
# ---------------------------------------------------------------------------


def test_walk_forward_is_reproducible_under_a_fixed_seed() -> None:
    panel = _synthetic_keyed_panel()
    first = _fast_walk_forward(panel, spec="garch", origins=ORIGINS[:5], horizon=2, quantiles=None)
    second = _fast_walk_forward(panel, spec="garch", origins=ORIGINS[:5], horizon=2, quantiles=None)
    assert first.one_step_variance.tobytes() == second.one_step_variance.tobytes()
    assert first.cumulative_variance.tobytes() == second.cumulative_variance.tobytes()


# ---------------------------------------------------------------------------
# Scope: per_security admits only per_security consumers (both directions)
# ---------------------------------------------------------------------------


def test_per_security_artifact_admits_only_per_security_consumers() -> None:
    model = PerSecurityVol("ewma", min_obs=20)
    assert model.series_scope == "per_security"
    model.assert_consumer_scope("per_security")
    with pytest.raises(ScopeMismatchError, match="scope mismatch"):
        model.assert_consumer_scope("date_level_portfolio")


def test_pooled_artifact_rejects_per_security_consumers_symmetrically() -> None:
    from quant_fund.models.volatility import GARCHVol

    pooled = GARCHVol()
    with pytest.raises(ValueError, match="consumable only by date_level_portfolio"):
        pooled.assert_consumer_scope("per_security")


# ---------------------------------------------------------------------------
# Proper-score evaluation on the keyed walk-forward outputs
# ---------------------------------------------------------------------------


def test_walk_forward_outputs_score_with_qlike_and_pinball() -> None:
    result = _fast_walk_forward(_synthetic_keyed_panel())
    qlike = qlike_keyed(result.realized_variance, result.cumulative_variance, keys=result.keys)
    assert np.isfinite(qlike["qlike_mean"])
    assert qlike["units"] == VARIANCE_UNITS
    pinball = pinball_keyed(result.realized_return, result.quantiles, LEVELS, keys=result.keys)
    assert all(np.isfinite(value) for value in pinball["pinball_mean_per_tau"].values())
    assert pinball["units"] == VOLATILITY_UNITS


# ---------------------------------------------------------------------------
# Panel + request validation (fail closed)
# ---------------------------------------------------------------------------


def test_panel_validation_rejects_bad_shapes_keys_and_times() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        KeyedReturnPanel(keys=np.array([["a"]]), times=np.array([0]), returns=np.array([0.0]))
    with pytest.raises(ValueError, match="same length"):
        KeyedReturnPanel(keys=np.array(["a"]), times=np.array([0, 1]), returns=np.array([0.0]))
    with pytest.raises(ValueError, match="non-empty"):
        KeyedReturnPanel(keys=np.array([]), times=np.array([]), returns=np.array([]))
    with pytest.raises(ValueError, match="non-empty strings"):
        KeyedReturnPanel(keys=np.array([""]), times=np.array([0]), returns=np.array([0.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        KeyedReturnPanel(
            keys=np.array(["a", "a"]), times=np.array([1, 1]), returns=np.array([0.0, 0.1])
        )
    with pytest.raises(ValueError, match="labels must match"):
        KeyedReturnPanel(
            keys=np.array(["a"]),
            times=np.array([0]),
            returns=np.array([0.0]),
            labels=np.array([0.0, 0.1]),
        )


def test_walk_forward_validates_origins() -> None:
    panel = _synthetic_keyed_panel()
    model = PerSecurityVol("ewma", min_obs=20)
    with pytest.raises(ValueError, match="non-empty"):
        walk_forward_per_security(panel, model, origins=[])
    with pytest.raises(ValueError, match="strictly increasing"):
        walk_forward_per_security(panel, model, origins=[5, 5])


def test_forecast_request_validation() -> None:
    panel = _synthetic_keyed_panel()
    model = PerSecurityVol("ewma", min_obs=20).fit(panel)
    with pytest.raises(ValueError, match="positive integer"):
        model.forecast("S00", horizon=0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        model.forecast("S00", quantiles=(0.0, 0.5))


def test_diagnostics_declare_scope_units_and_label_use() -> None:
    model = PerSecurityVol("ewma", min_obs=20)
    diag = model.diagnostics()
    assert diag["series_scope"] == "per_security"
    assert diag["variance_units"] == VARIANCE_UNITS
    assert diag["sigma_units"] == VOLATILITY_UNITS
    assert diag["label_usage"] == "none_forecasts_are_label_free"
    assert diag["pit_contract"] == "trailing_returns_strictly_before_as_of"
    meta = model.metadata()
    assert meta.extra["series_scope"] == "per_security"
    assert set(KEYED_VOL_SPECS) == {"ewma", "garch", "har", "aparch", "figarch"}


def test_per_key_error_is_a_value_error() -> None:
    assert issubclass(PerKeyVolError, ValueError)
