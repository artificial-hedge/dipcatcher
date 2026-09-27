"""EnbPI: ensemble batch prediction intervals for time series (Xu & Xie).

EnbPI wraps a point-prediction algorithm with block-bootstrap ensemble
"leave-one-out" (LOO) calibration and a sliding residual store, producing
sequential, distribution-free prediction intervals without data splitting
or retraining (Xu & Xie, 2021, ICML 38, PMLR 139:11559-11569,
"Conformal prediction interval for dynamic time-series"; journal extension
Xu & Xie, 2023, IEEE TPAMI 45(10):11575-11587, "Conformal prediction for
time series", arXiv:2010.09107, Algorithm 1).

Training phase (paper Algorithm 1, lines 1-10):
  1. Draw B bootstrap index sets S_1..S_B of the training series with a
     moving block bootstrap (blocks of length ``block_length``; default
     ceil(sqrt(T)) as in the paper's block-bootstrap comment; circular
     block wrap, seeded via ``np.random.default_rng``).
  2. Fit B models f^b on the bootstrap resamples (``base_model`` is a
     FACTORY returning a fresh sklearn-style estimator; default Ridge).
  3. For each training time t, the LOO ensemble predictor aggregates only
     models that did NOT include t in S_b: f_{-t}(x_t) = mean of f^b(x_t)
     over {b : t not in S_b} (paper line 7). If every model saw t (rare;
     probability ~ (1-(1-1/T)^T)^B), fall back to the full ensemble.
  4. Residual store: signed LOO residuals eps_t = y_t - f_{-t}(x_t)
     (paper lines 8-9).

Online phase (paper Algorithm 1, lines 11-22, batch size s=1):
  * Point forecast f(x_t) = mean over ALL B fitted models (paper line 12).
  * For a two-level request (tau_lo, tau_hi) the interval offsets are the
    width-minimizing contiguous residual quantile band (paper line 13:
    beta-hat = argmin_{beta in [0, 1-(tau_hi-tau_lo)]}
    [Q_{beta+tau_hi-tau_lo}(eps) - Q_beta(eps)], lines 14-16). For grids
    with more than two levels, offsets are the plain residual quantiles at
    each requested level (the beta-hat line search is defined only for a
    single central band).
  * ADAPTIVE phi_t correction: a multiplicative width scale with
    log-linear update driven by recent coverage errors,

        miss_t = 1{y_t not in [grid_lo, grid_hi]},
        phi_{t+1} = clip(phi_t * exp(gamma * (miss_t - alpha_band)),
                         PHI_MIN, PHI_MAX),

    where alpha_band = 1 - (tau_hi - tau_lo). DEVIATION (documented):
    neither the ICML 2021 paper nor the TPAMI 2023 extension (arXiv HTML
    and the authors' reference implementation, github.com/hamrel-cxu/EnbPI,
    both branches, checked 2026-09) defines a phi_t recursion; the paper's
    adaptivity is the sliding residual store plus the beta-hat width line
    search. The phi_t update above transplants the adaptive-conformal
    miscoverage recursion of Gibbs & Candes (2021, NeurIPS 34:1660-1672)
    onto the width scale (equivalently an integral controller on
    log-width; cf. Angelopoulos, Bates, Candes, Jordan, Zrnic, 2023,
    "Conformal PID control", arXiv:2310.16828). It is disabled (phi frozen
    at ``phi_init``) by passing ``adaptive=False``; that frozen variant is
    the ablation tested in tests/unit/test_enbpi.py.
  * Residual store slides forward (FIFO: drop oldest, append the new
    residual y_t - f(x_t)) every step (paper lines 17-22 with s=1), which
    is the paper's mechanism for adapting widths without retraining.
  * Output grids are made non-crossing by row-wise rearrangement
    (Chernozhukov, Fernandez-Val, Galichon, 2010, Ann. Statist. 38;
    quant_fund.metrics.scoring.rearrange_quantiles).

Honesty: only coverage, interval-width, and forecast diagnostics are
exposed (AGENTS.md honesty contract); no Sharpe/Sortino/P&L content.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from quant_fund.metrics.scoring import rearrange_quantiles

Array = NDArray[np.float64]

__all__ = ["EnbPI", "coverage"]

_PHI_BOUNDS: tuple[float, float] = (0.25, 4.0)
_BETA_SEARCH_GRID = 200


def coverage(grid: Array, y: float) -> float:
    """Containment indicator of scalar ``y`` in the extreme band of ``grid``.

    ``grid`` is a non-crossing quantile grid (increasing entries); the band
    is [grid[0], grid[-1]]. Returns 1.0 if contained, 0.0 otherwise.
    """
    g = np.asarray(grid, dtype=float).ravel()
    if g.size == 0 or not bool(np.all(np.isfinite(g))):
        raise ValueError("grid must be nonempty and finite")
    yv = float(y)
    if not np.isfinite(yv):
        raise ValueError("y must be finite")
    return float(g[0] <= yv <= g[-1])


class EnbPI:
    """EnbPI: ensemble batch prediction intervals for a time series.

    Parameters
    ----------
    quantile_levels:
        Strictly increasing levels in (0, 1). With exactly two levels the
        interval uses the paper's width-minimizing band (beta-hat line
        search); with more levels, plain residual quantiles per level.
    n_bootstraps:
        Number B of bootstrap models; B >= 1.
    block_length:
        Moving-block bootstrap block length; None -> ceil(sqrt(T)).
    base_model:
        Factory returning a fresh sklearn-style estimator (with ``fit`` and
        ``predict``); default ``Ridge()``.
    phi_init:
        Initial value of the adaptive width scale phi_t; must be positive.
        Frozen mode (``adaptive=False``) keeps phi_t == phi_init forever.
    random_state:
        Seed for ``np.random.default_rng`` (bootstrap resampling).
    lags:
        Autoregressive feature window used when ``x`` is not supplied
        (features are [y_{t-1}, ..., y_{t-lags}]).
    adaptive:
        True applies the phi_t coverage-error width update; False freezes
        phi_t at ``phi_init``.
    phi_learning_rate:
        gamma in the phi_t update; must be positive.

    Notes
    -----
    ``fit(y, x=None)`` runs the training phase; ``update(y, x=None)`` issues
    the conformalized grid for the current step (ensemble point forecast
    plus residual-quantile offsets scaled by phi_t) and then slides the
    residual store forward. Calling ``update`` before ``fit`` raises
    RuntimeError. All diagnostics are honesty-contract vocabulary only.
    """

    def __init__(
        self,
        quantile_levels: Array,
        n_bootstraps: int = 20,
        block_length: int | None = None,
        base_model: Callable[[], Any] | None = None,
        phi_init: float = 1.0,
        random_state: int | None = None,
        *,
        lags: int = 3,
        adaptive: bool = True,
        phi_learning_rate: float = 0.2,
    ) -> None:
        levels = np.asarray(quantile_levels, dtype=float).reshape(-1)
        if levels.size == 0:
            raise ValueError("quantile_levels must be non-empty")
        if not np.all(np.isfinite(levels)) or not np.all((levels > 0.0) & (levels < 1.0)):
            raise ValueError("quantile_levels must lie in (0, 1)")
        if np.any(np.diff(levels) <= 0.0):
            raise ValueError("quantile_levels must be strictly increasing")
        if int(n_bootstraps) < 1:
            raise ValueError("n_bootstraps must be >= 1")
        if block_length is not None and int(block_length) < 1:
            raise ValueError("block_length must be >= 1")
        phi0 = float(phi_init)
        if not np.isfinite(phi0) or phi0 <= 0.0:
            raise ValueError("phi_init must be positive and finite")
        if int(lags) < 1:
            raise ValueError("lags must be >= 1")
        gamma = float(phi_learning_rate)
        if not np.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("phi_learning_rate must be positive and finite")
        self._levels = levels
        self._n_boot = int(n_bootstraps)
        self._block_length = None if block_length is None else int(block_length)
        self._base_model = base_model if base_model is not None else Ridge
        self._lags = int(lags)
        self._adaptive = bool(adaptive)
        self._gamma = gamma
        self._phi = phi0
        self._rng = np.random.default_rng(random_state)
        self._models: list[Any] = []
        self._resid = np.empty(0, dtype=float)
        self._y_hist: list[float] = []
        self._n_feat = 0
        self._fitted = False
        self._phi_history: list[float] = []
        self._forecast_history: list[float] = []
        self.coverage_history: list[float] = []

    @property
    def quantile_levels_(self) -> Array:
        return self._levels.copy()

    @property
    def phi_(self) -> float:
        """Current value of the adaptive width scale phi_t."""
        return self._phi

    @property
    def phi_history(self) -> tuple[float, ...]:
        """phi_t after each online step (length n_steps)."""
        return tuple(self._phi_history)

    @property
    def forecast_history(self) -> Array:
        """Ensemble point forecasts issued at each online step."""
        return np.asarray(self._forecast_history, dtype=float)

    @property
    def residuals_(self) -> Array:
        """Current residual store (signed LOO residuals), a copy."""
        return self._resid.copy()

    @property
    def n_steps_(self) -> int:
        return len(self._forecast_history)

    def _ar_features(self, y: Array) -> Array:
        """Rows t = 0..T-1 of [y_{t-1}, ..., y_{t-lags}]; zero-padded early rows."""
        t = int(y.size)
        x = np.zeros((t, self._lags), dtype=float)
        for lag in range(1, self._lags + 1):
            x[lag:, lag - 1] = y[: t - lag]
        return x

    def fit(self, y: Array, x: Array | None = None) -> EnbPI:
        """Training phase: fit B bootstrap models and build the residual store.

        ``y`` is the 1-d training series (finite, length > lags). ``x`` is an
        optional (T, d) feature matrix; when None, autoregressive lag
        features of window ``lags`` are used.
        """
        yv = np.asarray(y, dtype=float).reshape(-1)
        if yv.size == 0:
            raise ValueError("training series must be non-empty")
        if not bool(np.all(np.isfinite(yv))):
            raise ValueError("training series must contain only finite values")
        if yv.size <= self._lags:
            raise ValueError(f"training series length must exceed lags={self._lags}")
        if x is None:
            feats = self._ar_features(yv)
        else:
            x_arr = np.asarray(x, dtype=float)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(-1, 1)
            feats = x_arr
            if feats.shape[0] != yv.size:
                raise ValueError("x must have one row per training time point")
            if not bool(np.all(np.isfinite(feats))):
                raise ValueError("x must contain only finite values")
        n = int(yv.size)
        t_fit = int(self._lags)
        x_fit = feats[t_fit:]
        y_fit = yv[t_fit:]
        block = self._block_length if self._block_length is not None else int(np.ceil(np.sqrt(n)))
        n_blocks = int(np.ceil(n / block))
        starts = self._rng.integers(0, n, size=n_blocks)
        offsets = np.arange(block, dtype=np.int64)
        boot_idx = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]

        preds = np.zeros((self._n_boot, n - t_fit), dtype=float)
        in_boot = np.zeros((self._n_boot, n), dtype=bool)
        self._models = []
        for b in range(self._n_boot):
            model = self._base_model()
            fit_rows = boot_idx[t_fit:] - t_fit
            model.fit(x_fit[fit_rows], y_fit[fit_rows])
            preds[b] = np.asarray(model.predict(x_fit), dtype=float).ravel()
            self._models.append(model)
            in_boot[b, boot_idx] = True

        resid = np.zeros(n - t_fit, dtype=float)
        for i in range(n - t_fit):
            t_abs = i + t_fit
            keep = np.flatnonzero(~in_boot[:, t_abs])
            if keep.size == 0:
                keep = np.arange(self._n_boot)
            resid[i] = y_fit[i] - float(np.mean(preds[keep, i]))
        self._resid = resid
        self._y_hist = [float(v) for v in yv]
        self._n_feat = int(feats.shape[1])
        self._fitted = True
        return self

    def _band_offsets(self, resid: Array) -> Array:
        """Residual-quantile offsets for the requested levels (phi applied later)."""
        if self._levels.size == 2:
            cov = float(self._levels[1] - self._levels[0])
            s = np.sort(resid)
            n = int(s.size)
            grid_pts = np.linspace(0.0, 1.0, n)
            betas = np.linspace(0.0, 1.0 - cov, _BETA_SEARCH_GRID + 1)
            upper = np.interp(betas + cov, grid_pts, s)
            lower = np.interp(betas, grid_pts, s)
            beta_hat = float(betas[int(np.argmin(upper - lower))])
            return np.array(
                [
                    np.interp(beta_hat, grid_pts, s),
                    np.interp(beta_hat + cov, grid_pts, s),
                ],
                dtype=float,
            )
        return np.asarray(np.quantile(resid, self._levels), dtype=float)

    def update(self, y: float, x: Array | None = None) -> Array:
        """Issue the conformalized grid for this step, then absorb ``y``.

        ``y`` is the realized value for the step being predicted; ``x`` is
        its feature vector (length n_features) or None for autoregressive
        lag features from the observed history. Returns the non-crossing
        conformalized quantile grid for the current step.
        """
        if not self._fitted:
            raise RuntimeError("EnbPI.update called before fit")
        y_arr = np.asarray(y, dtype=float)
        if y_arr.size != 1:
            raise ValueError("y must be a scalar")
        yv = float(y_arr.ravel()[0])
        if not np.isfinite(yv):
            raise ValueError("y must be finite")
        if x is None:
            row = np.asarray(
                [self._y_hist[-lag] for lag in range(1, self._lags + 1)], dtype=float
            ).reshape(1, -1)
        else:
            row = np.asarray(x, dtype=float).reshape(1, -1)
            if row.shape[1] != self._n_feat:
                raise ValueError(f"x must have {self._n_feat} features")
            if not bool(np.all(np.isfinite(row))):
                raise ValueError("x must contain only finite values")

        center = float(np.mean([m.predict(row)[0] for m in self._models]))
        offsets = self._band_offsets(self._resid)
        grid: Array = np.asarray(
            rearrange_quantiles((center + self._phi * offsets).reshape(1, -1)).ravel(),
            dtype=float,
        )
        contained = bool(grid[0] <= yv <= grid[-1])

        alpha_band = 1.0 - float(self._levels[-1] - self._levels[0])
        if self._adaptive:
            self._phi = float(
                np.clip(
                    self._phi * np.exp(self._gamma * (float(not contained) - alpha_band)),
                    _PHI_BOUNDS[0],
                    _PHI_BOUNDS[1],
                )
            )
        self._resid = np.concatenate([self._resid[1:], np.array([yv - center])])
        self._y_hist.append(yv)
        self._phi_history.append(self._phi)
        self._forecast_history.append(center)
        self.coverage_history.append(float(contained))
        return grid
