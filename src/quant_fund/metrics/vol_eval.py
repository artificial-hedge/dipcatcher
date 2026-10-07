"""Volatility forecast evaluation.

- ``mincer_zarnowitz``: MZ regression realized_t = a + b * forecast_t
  with OLS estimates, a joint Wald test of (a=0, b=1), and R2.
- Patton (2011) loss functions for variance forecasts given a noisy
  proxy x (e.g., squared return or realized variance): ``mse``,
  ``qlike`` (robust to proxy noise), ``mse_log``, ``hmse``, ``mae``.
- ``vol_loss_diff``: mean loss differential between two forecasts with
  a Newey-West standard error (a DM-ready pairwise comparison).

Fail-closed: non-positive forecasts for log-based losses, non-finite
input, length mismatch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from quant_fund.metrics.scoring import pinball_loss

Array = NDArray[np.float64]


def _check(x: Array, f: Array) -> tuple[Array, Array]:
    xx = np.asarray(x, dtype=float).ravel()
    ff = np.asarray(f, dtype=float).ravel()
    if xx.size != ff.size or xx.size < 10:
        raise ValueError("x and f must match, >= 10 obs")
    if not np.isfinite(xx).all() or not np.isfinite(ff).all():
        raise ValueError("non-finite input")
    return xx, ff


def mincer_zarnowitz(x: Array, f: Array) -> dict[str, float]:
    """MZ efficiency regression x_t = a + b f_t + e_t.

    Returns a, b, their standard errors, joint Wald chi2(2) stat for
    H0: a=0 & b=1, its p-value, and R2.
    """
    xx, ff = _check(x, f)
    n = xx.size
    xmat = np.column_stack([np.ones(n), ff])
    coef, *_ = np.linalg.lstsq(xmat, xx, rcond=None)
    resid = xx - xmat @ coef
    s2 = float(resid @ resid / (n - 2))
    cov_b = s2 * np.linalg.inv(xmat.T @ xmat)
    se = np.sqrt(np.maximum(np.diag(cov_b), 0.0))
    # joint Wald: R coef = (0, 1)
    rvec = coef - np.array([0.0, 1.0])
    try:
        wald = float(rvec @ np.linalg.inv(cov_b) @ rvec)
    except np.linalg.LinAlgError:
        wald = np.nan
    p = float(1.0 - stats.chi2.cdf(wald, 2)) if np.isfinite(wald) else np.nan
    ss_res = float(resid @ resid)
    ss_tot = float(((xx - xx.mean()) ** 2).sum())
    return {
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "se_alpha": float(se[0]),
        "se_beta": float(se[1]),
        "wald": wald,
        "wald_p": p,
        "r2": 1.0 - ss_res / max(ss_tot, 1e-14),
    }


def mse(x: Array, f: Array) -> Array:
    xx, ff = _check(x, f)
    return (xx - ff) ** 2


def qlike(x: Array, f: Array) -> Array:
    """Patton-robust QLIKE: x/f - ln(x/f) - 1 (needs x, f > 0)."""
    xx, ff = _check(x, f)
    if (ff <= 0).any() or (xx <= 0).any():
        raise ValueError("qlike requires positive forecasts and proxies")
    r = xx / ff
    return r - np.log(r) - 1.0


def mse_log(x: Array, f: Array) -> Array:
    xx, ff = _check(x, f)
    if (ff <= 0).any() or (xx <= 0).any():
        raise ValueError("mse_log requires positive inputs")
    return (np.log(xx) - np.log(ff)) ** 2


def hmse(x: Array, f: Array) -> Array:
    """Heteroskedasticity-adjusted MSE: (1 - x/f)^2 (Bollerslev-Ghysels)."""
    xx, ff = _check(x, f)
    if (ff <= 0).any():
        raise ValueError("hmse requires positive forecasts")
    return (1.0 - xx / ff) ** 2


def mae(x: Array, f: Array) -> Array:
    xx, ff = _check(x, f)
    return np.abs(xx - ff)


_LOSSES = {"mse": mse, "qlike": qlike, "mse_log": mse_log, "hmse": hmse, "mae": mae}


def vol_loss_diff(
    x: Array, f1: Array, f2: Array, loss: str = "qlike", max_lag: int | None = None
) -> dict[str, float]:
    """Mean loss differential d_t = L(x, f1) - L(x, f2) with NW se.

    Positive diff means f2 is better. Returns mean diff, se, t-stat.
    """
    if loss not in _LOSSES:
        raise ValueError(f"loss must be one of {sorted(_LOSSES)}")
    l1 = _LOSSES[loss](x, f1)
    l2 = _LOSSES[loss](x, f2)
    d = l1 - l2
    n = d.size
    if max_lag is None:
        max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    dc = d - d.mean()
    s = float(dc @ dc) / n
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        s += 2.0 * w * float(dc[lag:] @ dc[:-lag]) / n
    se = np.sqrt(max(s / n, 0.0))
    return {
        "mean_diff": float(d.mean()),
        "se": float(se),
        "t": float(d.mean() / max(se, 1e-14)),
        "n": float(n),
    }


# ---------------------------------------------------------------------------
# Units contract and keyed (per-security) proper-score evaluation.
#
# UNITS: ``VARIANCE_UNITS`` values are conditional VARIANCES (sigma^2) of
# decimal log returns; ``VOLATILITY_UNITS`` values are conditional standard
# deviations (sigma) of decimal log returns (the same numeric scale as decimal
# returns).  A variance stays a variance and a volatility stays a volatility:
# no function here ever converts between the two implicitly.  The ONLY
# sanctioned conversion is the explicit ``rescale_units``.
#
# The keyed scores below are PROPER SCORES ONLY (Patton QLIKE on variance
# forecasts, pinball on quantile forecasts).  Nothing here computes, reports or
# implies Sharpe/Sortino/Calmar/P&L/NAV or any live-performance claim; seeded
# SYNTHETIC panels scored with these functions are correctness tests only and
# are never market evidence.
# ---------------------------------------------------------------------------

VARIANCE_UNITS: Final[str] = "decimal_squared"
VOLATILITY_UNITS: Final[str] = "decimal"
KNOWN_VOL_UNITS: Final[frozenset[str]] = frozenset({VARIANCE_UNITS, VOLATILITY_UNITS})

_MIN_KEYED_OBS: Final[int] = 10


class VolUnitsError(ValueError):
    """Raised instead of ever silently rescaling variance <-> volatility."""


def _require_units_token(units: object) -> str:
    """Return the exact registered units token or fail closed."""
    if not isinstance(units, str) or not units.strip():
        raise VolUnitsError("units must be a non-empty string")
    token = units.strip()
    if token not in KNOWN_VOL_UNITS:
        raise VolUnitsError(f"unknown units {units!r}; known units: {sorted(KNOWN_VOL_UNITS)}")
    return token


def assert_units(actual: object, expected: object) -> str:
    """Require ``actual`` units to equal ``expected`` units exactly."""
    want = _require_units_token(expected)
    got = _require_units_token(actual)
    if got != want:
        raise VolUnitsError(
            f"units mismatch: got {got!r}, expected {want!r}; "
            "variance/volatility are never silently rescaled"
        )
    return got


def coerce_units(values: Array, *, from_units: object, to_units: object) -> NDArray[np.float64]:
    """Return ``values`` UNCHANGED (bit-identical) when the units match.

    Variance in -> variance out and volatility in -> volatility out with no
    silent rescale of any kind; a units mismatch raises ``VolUnitsError``.
    Use ``rescale_units`` for an explicit, documented sqrt/square conversion.
    """
    assert_units(from_units, to_units)
    return np.asarray(values, dtype=float)


def rescale_units(values: Array, *, from_units: object, to_units: object) -> NDArray[np.float64]:
    """EXPLICIT variance <-> volatility conversion (the only sanctioned one).

    Variance -> volatility takes the square root; volatility -> variance
    squares.  Same-units calls return the values unchanged.  Negative variance
    inputs rescale to NaN (honest) rather than to a silent complex or clipped
    value.
    """
    source = _require_units_token(from_units)
    target = _require_units_token(to_units)
    array = np.asarray(values, dtype=float)
    if source == target:
        return array
    with np.errstate(invalid="ignore"):
        return np.sqrt(array) if source == VARIANCE_UNITS else np.square(array)


def _keyed_matrix(name: str, values: Array, keys: Sequence[str]) -> NDArray[np.float64]:
    """Validate a (n_keys, n_obs) matrix against the explicit key axis."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-d (n_keys, n_obs) matrix")
    if array.shape[0] != len(keys):
        raise ValueError(f"{name} rows must match the {len(keys)} keys")
    if len(set(keys)) != len(keys):
        raise ValueError("keys must be unique")
    return array


def _keyed_quantile_tensor(
    name: str, values: Array, keys: Sequence[str], n_levels: int
) -> NDArray[np.float64]:
    """Validate a (n_keys, n_obs, n_levels) quantile tensor against the keys."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3-d (n_keys, n_obs, n_levels) tensor")
    if array.shape[0] != len(keys):
        raise ValueError(f"{name} rows must match the {len(keys)} keys")
    if array.shape[2] != n_levels:
        raise ValueError(f"{name} levels must match the {n_levels} taus")
    return array


def _mean_loss_or_nan(losses: NDArray[np.float64]) -> float:
    """Mean loss over scored rows; empty rows score honest NaN."""
    if losses.size == 0:
        return float("nan")
    return float(np.mean(losses))


def _qlike_row(realized: Array, forecast: Array) -> tuple[float, int]:
    """QLIKE mean for one key on its finite, positive pairs (honest NaN)."""
    mask = np.isfinite(realized) & np.isfinite(forecast) & (realized > 0.0) & (forecast > 0.0)
    n_scored = int(mask.sum())
    if n_scored < _MIN_KEYED_OBS:
        return float("nan"), n_scored
    losses = qlike(realized[mask], forecast[mask])
    return _mean_loss_or_nan(np.asarray(losses, dtype=float)), n_scored


def qlike_keyed(
    realized_var: Array,
    forecast_var: Array,
    *,
    keys: Sequence[str],
    units: str = VARIANCE_UNITS,
) -> dict[str, Any]:
    """Per-key Patton (2011) QLIKE for keyed variance forecasts.

    Both matrices are (n_keys, n_obs) and must be in ``VARIANCE_UNITS``
    (decimal-squared variance).  Volatility-unit inputs raise ``VolUnitsError``
    — a volatility is never silently squared into a variance.  Inputs pass
    through unchanged (variance in == variance out).  Rows with fewer than 10
    finite positive pairs score honest NaN for that key WITHOUT poisoning the
    other keys; the pooled ``qlike_mean`` uses every valid pair.

    Proper score only — no P&L/Sharpe/NAV claim of any kind.
    """
    assert_units(units, VARIANCE_UNITS)
    realized = _keyed_matrix("realized_var", realized_var, keys)
    forecast = _keyed_matrix("forecast_var", forecast_var, keys)
    per_key: dict[str, float] = {}
    counts: dict[str, int] = {}
    for index, key in enumerate(keys):
        per_key[key], counts[key] = _qlike_row(realized[index], forecast[index])
    pooled_realized = realized.reshape(-1)
    pooled_forecast = forecast.reshape(-1)
    pooled_mean, pooled_count = _qlike_row(pooled_realized, pooled_forecast)
    return {
        "score": "qlike",
        "units": units,
        "keys": list(keys),
        "qlike_per_key": per_key,
        "qlike_mean": pooled_mean,
        "n_scored_per_key": counts,
        "n_scored": pooled_count,
    }


def _pinball_row(realized: Array, quantile_row: Array, tau: float) -> tuple[float, int]:
    """Mean pinball for one key at one tau on its finite pairs (honest NaN)."""
    column = np.asarray(quantile_row, dtype=float)
    mask = np.isfinite(realized) & np.isfinite(column)
    n_scored = int(mask.sum())
    if n_scored < _MIN_KEYED_OBS:
        return float("nan"), n_scored
    losses = pinball_loss(realized[mask], column[mask], tau)
    return _mean_loss_or_nan(np.asarray(losses, dtype=float)), n_scored


def _validate_taus(taus: Sequence[float]) -> tuple[float, ...]:
    levels = tuple(float(tau) for tau in taus)
    if not levels or any(not 0.0 < tau < 1.0 for tau in levels):
        raise ValueError("taus must be non-empty and strictly between 0 and 1")
    return levels


def pinball_keyed(
    realized: Array,
    quantile_forecast: Array,
    taus: Sequence[float],
    *,
    keys: Sequence[str],
    units: str = VOLATILITY_UNITS,
) -> dict[str, Any]:
    """Per-key pinball scores for keyed quantile forecasts.

    ``realized`` is (n_keys, n_obs) and ``quantile_forecast`` is
    (n_keys, n_obs, n_levels) with ``levels == taus``; both are in the SAME
    declared ``units`` (decimal return quantiles use ``VOLATILITY_UNITS``;
    variance quantiles use ``VARIANCE_UNITS``).  Inputs pass through unchanged
    (a volatility in == a volatility out, a variance in == a variance out); the
    only sanctioned unit conversion is ``rescale_units``.  Keys with fewer than
    10 finite pairs at a tau score honest NaN WITHOUT poisoning other keys.

    Proper score only — no P&L/Sharpe/NAV claim of any kind.
    """
    levels = _validate_taus(taus)
    _require_units_token(units)
    realized = _keyed_matrix("realized", realized, keys)
    quantiles = _keyed_quantile_tensor("quantile_forecast", quantile_forecast, keys, len(levels))
    per_key: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for index, key in enumerate(keys):
        per_key[key], counts[key] = _pinball_key_all_taus(realized[index], quantiles[index], levels)
    pooled = _pooled_pinball_per_tau(realized, quantiles, levels)
    return {
        "score": "pinball",
        "units": units,
        "keys": list(keys),
        "taus": list(levels),
        "pinball_per_key": per_key,
        "pinball_mean_per_tau": pooled,
        "n_scored_per_key": counts,
    }


def _pinball_key_all_taus(
    realized_row: Array, quantile_rows: Array, levels: tuple[float, ...]
) -> tuple[dict[str, float], int]:
    """Pinball means for one key at every tau (honest NaN per tau)."""
    means: dict[str, float] = {}
    n_scored = 0
    for column, tau in enumerate(levels):
        means[_tau_key(tau)], n_scored = _pinball_row(realized_row, quantile_rows[:, column], tau)
    return means, n_scored


def _pooled_pinball_per_tau(
    realized: Array, quantiles: Array, levels: tuple[float, ...]
) -> dict[str, float]:
    """Pooled pinball per tau over every valid (key, obs) pair."""
    pooled: dict[str, float] = {}
    for column, tau in enumerate(levels):
        mean, _ = _pinball_row(realized.reshape(-1), quantiles[:, :, column].reshape(-1), tau)
        pooled[_tau_key(tau)] = mean
    return pooled


def _tau_key(tau: float) -> str:
    """Stable string key for a quantile level in result mappings."""
    return f"{tau:g}"
