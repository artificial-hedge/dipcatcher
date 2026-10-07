"""Keyed (per-security) proper-score evaluation and the units contract.

EVERY dataset in this file is seeded SYNTHETIC data, generated in-test and
labeled ``source_label="SYNTHETIC"``.  These are correctness tests only; they
are never market evidence and make no live-trading or profitability claim.

Covered:
- QLIKE of per-key variance forecasts (proper score, Patton 2011);
- pinball of per-key quantile forecasts (proper score);
- units preservation: variance in == variance out, volatility in ==
  volatility out, no silent rescale (explicit ``rescale_units`` only);
- honest-NaN for a failing key WITHOUT poisoning the other keys;
- deterministic seeded reproducibility of the synthetic panels.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.scoring import pinball_loss
from quant_fund.metrics.vol_eval import (
    VARIANCE_UNITS,
    VOLATILITY_UNITS,
    VolUnitsError,
    assert_units,
    coerce_units,
    pinball_keyed,
    qlike,
    qlike_keyed,
    rescale_units,
)

SYNTHETIC = "SYNTHETIC"
KEYS = ("S00", "S01", "S02", "S03")
TAUS = (0.1, 0.5, 0.9)


def _synthetic_variance_panel(
    seed: int = 20261007, n_obs: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded SYNTHETIC keyed variance panel: realized labels + forecasts."""
    rng = np.random.default_rng(seed)
    realized = np.abs(rng.normal(1e-4, 2e-5, size=(len(KEYS), n_obs))) + 1e-6
    forecast = realized * np.exp(rng.normal(0.0, 0.1, size=realized.shape))
    assert SYNTHETIC == "SYNTHETIC"
    return realized, forecast


def _synthetic_quantile_panel(
    seed: int = 20261008, n_obs: int = 40
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded SYNTHETIC keyed return panel: realized + quantile forecasts."""
    rng = np.random.default_rng(seed)
    realized = rng.normal(0.0, 0.01, size=(len(KEYS), n_obs))
    center = realized * 0.8 + rng.normal(0.0, 0.002, size=realized.shape)
    spread = np.abs(rng.normal(0.01, 0.001, size=realized.shape))
    offsets = np.asarray([-1.28, 0.0, 1.28])
    quantiles = center[:, :, None] + spread[:, :, None] * offsets[None, None, :]
    return realized, quantiles, np.asarray(TAUS)


def test_qlike_keyed_scores_seeded_synthetic_per_key_variance_forecasts() -> None:
    realized, forecast = _synthetic_variance_panel()
    result = qlike_keyed(realized, forecast, keys=KEYS, units=VARIANCE_UNITS)

    assert result["score"] == "qlike"
    assert result["units"] == VARIANCE_UNITS
    assert result["keys"] == list(KEYS)
    assert np.isfinite(result["qlike_mean"])
    assert result["qlike_mean"] > 0.0
    for key in KEYS:
        assert np.isfinite(result["qlike_per_key"][key])
        assert result["n_scored_per_key"][key] == realized.shape[1]
        # Reusing the scalar Patton loss must agree row by row.
        expected = float(np.mean(qlike(realized[KEYS.index(key)], forecast[KEYS.index(key)])))
        assert result["qlike_per_key"][key] == pytest.approx(expected)


def test_qlike_keyed_matches_a_perfect_forecast_of_zero() -> None:
    values = np.full((len(KEYS), 12), 3e-4)
    result = qlike_keyed(values, values.copy(), keys=KEYS)
    assert result["qlike_mean"] == pytest.approx(0.0, abs=1e-12)


def test_qlike_keyed_rejects_volatility_units_without_squaring() -> None:
    realized, forecast = _synthetic_variance_panel()
    with pytest.raises(VolUnitsError, match="units mismatch"):
        qlike_keyed(realized, forecast, keys=KEYS, units=VOLATILITY_UNITS)


def test_qlike_keyed_passes_variance_in_as_variance_out_unchanged() -> None:
    realized, forecast = _synthetic_variance_panel()
    result = qlike_keyed(realized, forecast, keys=KEYS)
    assert result["units"] == VARIANCE_UNITS
    # Variance in == variance out: scoring never rescales its inputs.
    manual = [float(np.mean(qlike(realized[i], forecast[i]))) for i in range(realized.shape[0])]
    assert [result["qlike_per_key"][key] for key in KEYS] == pytest.approx(manual)


def test_qlike_keyed_one_bad_key_does_not_poison_the_others() -> None:
    realized, forecast = _synthetic_variance_panel()
    bad = 2
    forecast[bad, :] = np.nan  # SYNTHETIC per-key failure
    result = qlike_keyed(realized, forecast, keys=KEYS)

    assert np.isnan(result["qlike_per_key"][KEYS[bad]])
    assert result["n_scored_per_key"][KEYS[bad]] == 0
    for index, key in enumerate(KEYS):
        if index != bad:
            assert np.isfinite(result["qlike_per_key"][key])
    assert np.isfinite(result["qlike_mean"])


def test_qlike_keyed_short_key_scores_honest_nan() -> None:
    realized, forecast = _synthetic_variance_panel()
    forecast[0, 10:] = np.nan  # only 10 finite pairs -> under the floor with 9
    forecast[0, 9:] = np.nan
    result = qlike_keyed(realized, forecast, keys=KEYS)

    assert np.isnan(result["qlike_per_key"][KEYS[0]])
    assert result["n_scored_per_key"][KEYS[0]] == 9
    assert np.isfinite(result["qlike_per_key"][KEYS[1]])


def test_pinball_keyed_scores_seeded_synthetic_per_key_quantile_forecasts() -> None:
    realized, quantiles, taus = _synthetic_quantile_panel()
    result = pinball_keyed(realized, quantiles, taus, keys=KEYS, units=VOLATILITY_UNITS)

    assert result["score"] == "pinball"
    assert result["units"] == VOLATILITY_UNITS
    assert result["taus"] == list(taus)
    for key in KEYS:
        index = KEYS.index(key)
        for tau in taus:
            expected = float(
                np.mean(
                    pinball_loss(realized[index], quantiles[index, :, list(taus).index(tau)], tau)
                )
            )
            assert result["pinball_per_key"][key][f"{tau:g}"] == pytest.approx(expected)
            assert np.isfinite(result["pinball_mean_per_tau"][f"{tau:g}"])


def test_pinball_keyed_scores_variance_quantiles_in_variance_units() -> None:
    realized, quantiles, taus = _synthetic_quantile_panel(seed=31)
    variance_scale = 1e-4
    result = pinball_keyed(
        realized**2 / variance_scale,
        quantiles**2 / variance_scale,
        taus,
        keys=KEYS,
        units=VARIANCE_UNITS,
    )
    assert result["units"] == VARIANCE_UNITS
    assert all(np.isfinite(value) for value in result["pinball_mean_per_tau"].values())


def test_pinball_keyed_one_bad_key_does_not_poison_the_others() -> None:
    realized, quantiles, taus = _synthetic_quantile_panel()
    bad = 1
    quantiles[bad, :, :] = np.nan  # SYNTHETIC per-key failure
    result = pinball_keyed(realized, quantiles, taus, keys=KEYS)

    for tau in taus:
        assert np.isnan(result["pinball_per_key"][KEYS[bad]][f"{tau:g}"])
        assert np.isfinite(result["pinball_mean_per_tau"][f"{tau:g}"])
    assert np.isfinite(result["pinball_per_key"][KEYS[0]]["0.5"])


def test_pinball_keyed_rejects_bad_tau_and_shape_inputs() -> None:
    realized, quantiles, taus = _synthetic_quantile_panel()
    with pytest.raises(ValueError, match="taus"):
        pinball_keyed(realized, quantiles, (0.0, 0.5), keys=KEYS)
    with pytest.raises(ValueError, match="3-d"):
        pinball_keyed(realized, quantiles[0], taus, keys=KEYS)
    with pytest.raises(ValueError, match="rows must match"):
        pinball_keyed(realized, quantiles, taus, keys=KEYS[:-1])
    with pytest.raises(ValueError, match="unique"):
        qlike_keyed(realized**2, realized**2, keys=("S00", "S00", "S01", "S02"))


def test_qlike_keyed_rejects_non_matrix_inputs() -> None:
    values = np.ones(20)
    with pytest.raises(ValueError, match="2-d"):
        qlike_keyed(values, values, keys=KEYS)


def test_coerce_units_is_bit_identical_within_each_unit() -> None:
    rng = np.random.default_rng(99)  # SYNTHETIC
    variance = np.abs(rng.normal(1e-4, 1e-5, 32)) + 1e-9
    volatility = np.sqrt(variance)

    out_variance = coerce_units(variance, from_units=VARIANCE_UNITS, to_units=VARIANCE_UNITS)
    out_volatility = coerce_units(
        volatility, from_units=VOLATILITY_UNITS, to_units=VOLATILITY_UNITS
    )

    assert out_variance.tobytes() == variance.tobytes()
    assert out_volatility.tobytes() == volatility.tobytes()


def test_coerce_units_never_rescales_across_units() -> None:
    variance = np.full(4, 1e-4)
    with pytest.raises(VolUnitsError, match="never silently rescaled"):
        coerce_units(variance, from_units=VARIANCE_UNITS, to_units=VOLATILITY_UNITS)
    with pytest.raises(VolUnitsError, match="never silently rescaled"):
        coerce_units(variance, from_units=VOLATILITY_UNITS, to_units=VARIANCE_UNITS)


def test_rescale_units_is_the_only_explicit_conversion_and_round_trips() -> None:
    rng = np.random.default_rng(123)  # SYNTHETIC
    variance = np.abs(rng.normal(1e-4, 1e-5, 16)) + 1e-9

    volatility = rescale_units(variance, from_units=VARIANCE_UNITS, to_units=VOLATILITY_UNITS)
    back = rescale_units(volatility, from_units=VOLATILITY_UNITS, to_units=VARIANCE_UNITS)

    assert np.allclose(volatility, np.sqrt(variance))
    assert np.allclose(back, variance)
    same = rescale_units(variance, from_units=VARIANCE_UNITS, to_units=VARIANCE_UNITS)
    assert same.tobytes() == variance.tobytes()


def test_units_guards_fail_closed_on_missing_or_unknown_tokens() -> None:
    for bad in ("", "   ", "volatility", "decimal_cubed", None, 5):
        with pytest.raises(VolUnitsError):
            assert_units(bad, VARIANCE_UNITS)
        with pytest.raises(VolUnitsError):
            coerce_units(np.ones(3), from_units=VARIANCE_UNITS, to_units=bad)
    assert assert_units(VARIANCE_UNITS, "decimal_squared") == VARIANCE_UNITS


def test_keyed_panels_are_seeded_and_reproducible() -> None:
    first = _synthetic_variance_panel(seed=5)
    second = _synthetic_variance_panel(seed=5)
    assert first[0].tobytes() == second[0].tobytes()
    assert first[1].tobytes() == second[1].tobytes()
    result_first = qlike_keyed(first[0], first[1], keys=KEYS)
    result_second = qlike_keyed(second[0], second[1], keys=KEYS)
    assert result_first == result_second
