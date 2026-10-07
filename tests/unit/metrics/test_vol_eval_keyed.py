"""Keyed QLIKE/pinball proper scores on seeded SYNTHETIC keyed panels.

Honesty: synthetic data only (always labeled), proper scores only (Patton
QLIKE, pinball) — no Sharpe/Sortino/Calmar/P&L/NAV keys, no market-evidence
or profitability claim.  Keyed honesty: one bad key is NaN without poisoning
its siblings.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.vol_eval import (
    VARIANCE_UNITS,
    VOLATILITY_UNITS,
    VolUnitsError,
    assert_units,
    coerce_units,
    pinball_keyed,
    qlike_keyed,
    rescale_units,
)

KEYS = ("S00", "S01", "S02", "S03")
TAUS = (0.1, 0.5, 0.9)
SOURCE_LABEL = "SYNTHETIC"
_SEED = 20261007
N_OBS = 40


def _synthetic_pair(seed: int = _SEED, n_keys: int = 4, n_obs: int = N_OBS):
    """Seeded SYNTHETIC realized/forecast VARIANCE matrices (n_keys, n_obs)."""
    rng = np.random.default_rng(seed)
    realized = np.exp(rng.normal(-8.0, 0.5, size=(n_keys, n_obs)))
    forecast = realized * np.exp(rng.normal(0.0, 0.2, size=(n_keys, n_obs)))
    return realized, forecast


def _synthetic_quantiles(seed: int = _SEED, n_keys: int = 4, n_obs: int = N_OBS):
    """Seeded SYNTHETIC (returns, quantile surfaces) for pinball scoring."""
    rng = np.random.default_rng(seed)
    realized = rng.normal(0.0, 0.01, size=(n_keys, n_obs))
    base = np.quantile(rng.normal(0.0, 0.01, size=(n_keys, 1000)), TAUS, axis=1).T
    quantiles = np.repeat(base[:, None, :], n_obs, axis=1)
    quantiles += rng.normal(0.0, 0.001, size=quantiles.shape)
    return realized, quantiles


def test_qlike_keyed_scores_on_synthetic_variances() -> None:
    realized, forecast = _synthetic_pair()
    out = qlike_keyed(realized, forecast, keys=KEYS)
    assert out["score"] == "qlike"
    assert out["units"] == VARIANCE_UNITS
    assert SOURCE_LABEL == "SYNTHETIC"
    assert np.isfinite(out["qlike_mean"])
    assert set(out["qlike_per_key"]) == set(KEYS)
    assert all(np.isfinite(score) for score in out["qlike_per_key"].values())
    assert out["n_scored"] == 4 * N_OBS


def test_qlike_is_decentralized_over_keys() -> None:
    realized, forecast = _synthetic_pair()
    pooled = qlike_keyed(realized, forecast, keys=KEYS)
    for index, key in enumerate(KEYS):
        single = qlike_keyed(realized[index : index + 1], forecast[index : index + 1], keys=(key,))
        assert np.isclose(single["qlike_per_key"][key], pooled["qlike_per_key"][key])


def test_qlike_rejects_volatility_units_explicitly() -> None:
    realized, forecast = _synthetic_pair()
    with pytest.raises(VolUnitsError):
        qlike_keyed(realized, forecast, keys=KEYS, units=VOLATILITY_UNITS)


def test_qlike_bad_key_is_honest_nan_without_poisoning_siblings() -> None:
    realized, forecast = _synthetic_pair()
    realized[2, :] = np.nan  # key S02 has zero finite pairs
    out = qlike_keyed(realized, forecast, keys=KEYS)
    assert np.isnan(out["qlike_per_key"]["S02"])
    for key in ("S00", "S01", "S03"):
        assert np.isfinite(out["qlike_per_key"][key])
    assert np.isfinite(out["qlike_mean"])
    assert out["n_scored_per_key"]["S02"] == 0


def test_qlike_short_key_is_honest_nan() -> None:
    realized, forecast = _synthetic_pair()
    realized[1, 10:] = np.nan  # only 10 finite pairs remains? make it 5
    realized[1, 5:] = np.nan
    out = qlike_keyed(realized, forecast, keys=KEYS)
    assert np.isnan(out["qlike_per_key"]["S01"])
    assert out["n_scored_per_key"]["S01"] == 5
    assert np.isfinite(out["qlike_per_key"]["S00"])


def test_qlike_rejects_duplicate_keys_and_shapes() -> None:
    realized, forecast = _synthetic_pair()
    with pytest.raises(ValueError, match="unique"):
        qlike_keyed(realized, forecast, keys=("S00", "S00", "S01", "S02"))
    with pytest.raises(ValueError, match="n_keys"):
        qlike_keyed(realized[:2], forecast, keys=KEYS)


def test_pinball_keyed_scores_on_synthetic_quantiles() -> None:
    realized, quantiles = _synthetic_quantiles()
    out = pinball_keyed(realized, quantiles, TAUS, keys=KEYS)
    assert out["score"] == "pinball"
    assert out["units"] == VOLATILITY_UNITS
    assert out["taus"] == TAUS
    assert all(np.isfinite(value) for value in out["pinball_mean_per_tau"].values())
    for key in KEYS:
        assert all(np.isfinite(score) for score in out["pinball_per_key"][key].values())


def test_pinball_per_key_is_decentralized() -> None:
    realized, quantiles = _synthetic_quantiles()
    pooled = pinball_keyed(realized, quantiles, TAUS, keys=KEYS)
    for index, key in enumerate(KEYS):
        single = pinball_keyed(
            realized[index : index + 1], quantiles[index : index + 1], TAUS, keys=(key,)
        )
        for tau in TAUS:
            assert np.isclose(single["pinball_per_key"][key][str(tau)], pooled["pinball_per_key"][key][str(tau)])


def test_pinball_bad_key_is_honest_nan_without_poisoning_siblings() -> None:
    realized, quantiles = _synthetic_quantiles()
    realized[2, :] = np.nan
    out = pinball_keyed(realized, quantiles, TAUS, keys=KEYS)
    assert all(np.isnan(score) for score in out["pinball_per_key"]["S02"].values())
    for key in ("S00", "S01", "S03"):
        assert all(np.isfinite(score) for score in out["pinball_per_key"][key].values())
    assert all(np.isfinite(value) for value in out["pinball_mean_per_tau"].values())


def test_pinball_short_key_is_honest_nan() -> None:
    realized, quantiles = _synthetic_quantiles()
    realized[1, 5:] = np.nan
    out = pinball_keyed(realized, quantiles, TAUS, keys=KEYS)
    assert all(np.isnan(score) for score in out["pinball_per_key"]["S01"].values())
    assert np.isfinite(out["pinball_per_key"]["S00"][str(TAUS[0])])


def test_pinball_requires_matching_levels_and_valid_taus() -> None:
    realized, quantiles = _synthetic_quantiles()
    with pytest.raises(ValueError, match="n_levels"):
        pinball_keyed(realized, quantiles[:, :, :2], TAUS, keys=KEYS)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        pinball_keyed(realized, quantiles, (0.0, 0.5), keys=KEYS)


def test_units_contract_is_bit_preserving_and_fail_closed() -> None:
    variance = np.array([4e-4, 1e-4])
    sigma = np.array([0.02, 0.01])
    assert coerce_units(variance, from_units=VARIANCE_UNITS, to_units=VARIANCE_UNITS).tobytes() == variance.tobytes()
    assert coerce_units(sigma, from_units=VOLATILITY_UNITS, to_units=VOLATILITY_UNITS).tobytes() == sigma.tobytes()
    with pytest.raises(VolUnitsError):
        coerce_units(variance, from_units=VARIANCE_UNITS, to_units=VOLATILITY_UNITS)
    assert assert_units(VARIANCE_UNITS, VARIANCE_UNITS) == VARIANCE_UNITS
    with pytest.raises(VolUnitsError):
        assert_units(VARIANCE_UNITS, VOLATILITY_UNITS)
    with pytest.raises(VolUnitsError):
        assert_units("variance_squared", VARIANCE_UNITS)


def test_rescale_units_round_trips_explicitly() -> None:
    variance = np.array([4e-4, 1e-4, 2.5e-4])
    sigma = rescale_units(variance, from_units=VARIANCE_UNITS, to_units=VOLATILITY_UNITS)
    assert np.allclose(sigma, np.sqrt(variance))
    back = rescale_units(sigma, from_units=VOLATILITY_UNITS, to_units=VARIANCE_UNITS)
    assert np.allclose(back, variance)


def test_keyed_scores_are_seeded_reproducible() -> None:
    realized, forecast = _synthetic_pair(seed=_SEED)
    realized2, forecast2 = _synthetic_pair(seed=_SEED)
    first = qlike_keyed(realized, forecast, keys=KEYS)
    second = qlike_keyed(realized2, forecast2, keys=KEYS)
    assert first["qlike_per_key"] == second["qlike_per_key"]
    realized_q, quantiles = _synthetic_quantiles(seed=_SEED)
    realized_q2, quantiles2 = _synthetic_quantiles(seed=_SEED)
    third = pinball_keyed(realized_q, quantiles, TAUS, keys=KEYS)
    fourth = pinball_keyed(realized_q2, quantiles2, TAUS, keys=KEYS)
    assert third["pinball_per_key"] == fourth["pinball_per_key"]
