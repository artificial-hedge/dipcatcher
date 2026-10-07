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
# Units contract and keyed proper scores (per-security volatility scope)
#
# UNITS CONTRACT: a VARIANCE (``VARIANCE_UNITS = "decimal_squared"``, sigma^2
# of decimal log returns) stays a variance and a VOLATILITY
# (``VOLATILITY_UNITS = "decimal"``, sigma on the same scale as returns) stays
# a volatility end to end.  ``coerce_units`` preserves values bit-identically
# within one unit and raises ``VolUnitsError`` across units — there is NO
# silent rescale.  ``rescale_units`` is the ONLY sanctioned conversion (an
# explicit sqrt/square at an API boundary that documents both sides).
#
# Scoring is proper-scores-only (Patton 2011 QLIKE; pinball) with per-key
# honesty: a key with too few finite scored pairs yields NaN for that key
# only, never poisoning sibling keys or the pooled mean.  Nothing here emits
# Sharpe/Sortino/Calmar/P&L/NAV keys (frozen honesty contract).
# ---------------------------------------------------------------------------

VARIANCE_UNITS: Final[str] = "decimal_squared"
VOLATILITY_UNITS: Final[str] = "decimal"
KNOWN_VOL_UNITS: Final[frozenset[str]] = frozenset({VARIANCE_UNITS, VOLATILITY_UNITS})
_MIN_KEYED_OBS: Final[int] = 10


class VolUnitsError(ValueError):
    """Raised when a value crosses unit families without ``rescale_units``."""


def _require_units_token(units: object) -> str:
    if not isinstance(units, str) or units not in KNOWN_VOL_UNITS:
        raise VolUnitsError(f"unknown vol units {units!r}; known: {sorted(KNOWN_VOL_UNITS)}")
    return units


def assert_units(actual: object, expected: object) -> str:
    """Require ``actual == expected`` units; return the validated token."""
    actual_token = _require_units_token(actual)
    expected_token = _require_units_token(expected)
    if actual_token != expected_token:
        raise VolUnitsError(
            f"units mismatch: got {actual_token!r}, expected {expected_token!r}; "
            "use rescale_units explicitly at the boundary"
        )
    return actual_token


def coerce_units(values: Array, *, from_units: object, to_units: object) -> Array:
    """Return ``values`` unchanged when units match (bit-identical), else raise.

    A variance in produces a variance out and a volatility in produces a
    volatility out with NO silent rescale.  Crossing unit families raises
    ``VolUnitsError``; only ``rescale_units`` performs a conversion.
    """
    source = _require_units_token(from_units)
    target = _require_units_token(to_units)
    if source != target:
        raise VolUnitsError(
            f"refusing to coerce {source!r} to {target!r} silently; call rescale_units"
        )
    return np.asarray(values, dtype=float)


def rescale_units(values: Array, *, from_units: object, to_units: object) -> Array:
    """The ONLY sanctioned variance<->volatility conversion (explicit sqrt/square)."""
    source = _require_units_token(from_units)
    target = _require_units_token(to_units)
    arr = np.asarray(values, dtype=float)
    if source == target:
        return arr
    if source == VARIANCE_UNITS:
        return np.asarray(np.sqrt(np.maximum(arr, 0.0)), dtype=np.float64)
    return np.asarray(arr * arr, dtype=np.float64)


def _keyed_matrix(values: Array, n_keys: int, name: str) -> Array:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != n_keys:
        raise ValueError(f"{name} must be (n_keys, n_obs) with n_keys == {n_keys}")
    return matrix


def _keyed_quantile_tensor(values: Array, n_keys: int, n_levels: int, name: str) -> Array:
    tensor = np.asarray(values, dtype=float)
    if tensor.ndim != 3 or tensor.shape[0] != n_keys or tensor.shape[2] != n_levels:
        raise ValueError(f"{name} must be (n_keys, n_obs, n_levels)")
    return tensor


def _qlike_row(realized: Array, forecast: Array) -> float:
    mask = np.isfinite(realized) & np.isfinite(forecast) & (realized > 0) & (forecast > 0)
    if int(mask.sum()) < _MIN_KEYED_OBS:
        return float("nan")
    return float(np.mean(qlike(realized[mask], forecast[mask])))


def qlike_keyed(
    realized_var: Array,
    forecast_var: Array,
    *,
    keys: Sequence[str],
    units: str = VARIANCE_UNITS,
) -> dict[str, Any]:
    """Per-key Patton QLIKE on VARIANCE inputs (proper score, keyed).

    ``realized_var`` and ``forecast_var`` are (n_keys, n_obs) matrices of
    realized variance and variance forecasts (NOT volatility).  QLIKE is not
    invariant to a square, so volatility-unit inputs raise ``VolUnitsError``
    instead of being silently squared.  Keys with fewer than 10 finite
    positive scored pairs are honest NaN without poisoning other keys.
    """
    assert_units(units, VARIANCE_UNITS)
    key_list = [str(key) for key in keys]
    if len(key_list) != len(set(key_list)):
        raise ValueError("keys must be unique")
    realized = _keyed_matrix(realized_var, len(key_list), "realized_var")
    forecast = _keyed_matrix(forecast_var, len(key_list), "forecast_var")
    per_key = {
        key: _qlike_row(realized[index], forecast[index])
        for index, key in enumerate(key_list)
    }
    scored = [value for value in per_key.values() if np.isfinite(value)]
    counts = {
        key: int(
            np.sum(
                np.isfinite(realized[index])
                & np.isfinite(forecast[index])
                & (realized[index] > 0)
                & (forecast[index] > 0)
            )
        )
        for index, key in enumerate(key_list)
    }
    return {
        "score": "qlike",
        "units": VARIANCE_UNITS,
        "keys": key_list,
        "qlike_per_key": per_key,
        "qlike_mean": float(np.mean(scored)) if scored else float("nan"),
        "n_scored_per_key": counts,
        "n_scored": int(sum(counts.values())),
    }


def _validate_taus(taus: Sequence[float]) -> tuple[float, ...]:
    levels = tuple(float(tau) for tau in taus)
    if not levels or any(not 0.0 < tau < 1.0 for tau in levels):
        raise ValueError("taus must lie strictly between 0 and 1")
    return levels


def _pinball_row(realized: Array, quantiles: Array, taus: tuple[float, ...]) -> dict[str, float]:
    row: dict[str, float] = {}
    for level_index, tau in enumerate(taus):
        pairs = np.column_stack([realized, quantiles[:, level_index]])
        finite = pairs[np.isfinite(pairs).all(axis=1)]
        row[str(tau)] = (
            float(np.mean(_pinball_key_all_taus(finite[:, 0], finite[:, 1], (tau,))[0]))
            if finite.shape[0] >= _MIN_KEYED_OBS
            else float("nan")
        )
    return row


def _pinball_key_all_taus(realized: Array, forecast: Array, taus: tuple[float, ...]) -> Array:
    """Pinball values per tau (vectorized over tau)."""
    return np.array([pinball_loss(realized, forecast, tau).mean() for tau in taus], dtype=float)


def pinball_keyed(
    realized: Array,
    quantile_forecast: Array,
    taus: Sequence[float],
    *,
    keys: Sequence[str],
    units: str = VOLATILITY_UNITS,
) -> dict[str, Any]:
    """Per-key pinball loss on predictive quantiles (proper score, keyed).

    ``realized`` is (n_keys, n_obs) and ``quantile_forecast`` is
    (n_keys, n_obs, n_levels) with one slice per requested tau.  Units must be
    declared explicitly (``VOLATILITY_UNITS`` for return quantiles, or
    ``VARIANCE_UNITS`` for variance quantiles); no rescale happens here.
    Keys with fewer than 10 finite scored pairs per tau are honest NaN without
    poisoning other keys.
    """
    _require_units_token(units)
    levels = _validate_taus(taus)
    key_list = [str(key) for key in keys]
    if len(key_list) != len(set(key_list)):
        raise ValueError("keys must be unique")
    realized_matrix = _keyed_matrix(realized, len(key_list), "realized")
    quantiles = _keyed_quantile_tensor(quantile_forecast, len(key_list), len(levels), "quantile_forecast")
    per_key = {
        key: _pinball_row(realized_matrix[index], quantiles[index], levels)
        for index, key in enumerate(key_list)
    }
    mean_per_tau = _pooled_pinball_per_tau(realized_matrix, quantiles, levels, key_list)
    counts = {
        key: int(np.isfinite(realized_matrix[index]).sum())
        for index, key in enumerate(key_list)
    }
    return {
        "score": "pinball",
        "units": _require_units_token(units),
        "keys": key_list,
        "taus": levels,
        "pinball_per_key": per_key,
        "pinball_mean_per_tau": mean_per_tau,
        "n_scored_per_key": counts,
    }


def _pooled_pinball_per_tau(
    realized: Array, quantiles: Array, levels: tuple[float, ...], keys: list[str]
) -> dict[str, float]:
    pooled: dict[str, float] = {}
    for level_index, tau in enumerate(levels):
        values = [
            per_tau
            for index, key in enumerate(keys)
            for per_tau in [_pinball_key_all_taus(realized[index], quantiles[index, :, level_index], (tau,))[0]]
            if np.isfinite(per_tau)
        ]
        pooled[str(tau)] = float(np.mean(values)) if values else float("nan")
    return pooled


def _tau_key(tau: float) -> str:
    return str(float(tau))
