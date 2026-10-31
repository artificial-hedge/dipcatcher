"""Multi-horizon EnbPI — per-horizon residual ensembles for y_{t+h}, h = 1..H.

Direct extension of EnbPI (Xu & Xie, 2021, ICML, PMLR 139, pp. 11559-11569;
2023, IEEE TPAMI 45(10), "Conformal prediction for time series",
arXiv:2010.09107) to multiple forecast horizons, reusing this repo's
``quant_fund.models.enbpi`` primitives (circular block bootstrap, LOO
ensemble residuals, narrowest signed-residual band).

Construction:

1. For each horizon ``h = 1..H``, train ``n_estimators`` bootstrap models
   ``f_b^h`` on circular block-bootstrap resamples of the pairs
   ``(x_i, y_{i+h})``, ``i = 0..n-h-1`` (the horizon-``h`` design).
2. Per-horizon LOO signed residuals ``eps_i^h = y_{i+h} - f^{-i,h}(x_i)``
   aggregate the models whose bootstrap sample excluded point ``i``, exactly
   as in EnbPI Algorithm 1 but per horizon.
3. At test time ``t``, the point forecast for horizon ``h`` aggregates all
   ``B`` models and the band adds the narrowest empirical
   ``[q_beta, q_{1-alpha_h+beta}]`` window of the horizon-``h`` residual
   window. Once ``y_{t+h}`` is revealed, the residual replaces the oldest
   entry of that horizon's window — updates are therefore *delayed by h
   steps*, and every prediction is issued strictly before its target label
   is observed.

Coverage accounting (honesty contract):

- Each horizon carries its own **marginal** approximate guarantee inherited
  from EnbPI's Theorem 1 (valid under strong mixing of that horizon's error
  sequence plus an ensemble-estimation error bound; not a finite-sample
  distribution-free guarantee).
- **Joint** coverage ``P(all H intervals cover at the same origin)`` is only
  controlled through the classic Bonferroni union bound
  ``>= 1 - sum_h alpha_h``. With ``bonferroni_joint=True`` each horizon is
  calibrated at ``alpha_h = alpha / H`` so the joint bound is ``1 - alpha``;
  otherwise ``alpha_h = alpha`` per horizon (joint bound ``1 - H * alpha``,
  clipped at 0). The bound is a worst case over dependence between horizons
  — the empirically realized joint coverage is typically higher because
  overlapping targets share innovations.

Fail-closed: bad hyperparameters, too-short series (per horizon we keep
EnbPI's ``max(10, ceil(1/alpha_h) + 1)`` floor), degenerate LOO sets, test
windows too short to evaluate joint coverage, and non-finite inputs all
raise. Querying an unfitted model raises ``RuntimeError``. Honesty: only
coverage / width diagnostics are exposed; SYNTHETIC tests of this module are
correctness tests, never market evidence; no live-trading claims.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.enbpi import (
    Regressor,
    circular_block_bootstrap_indices,
    enbpi_residual_bounds,
)

Array = NDArray[np.float64]

__all__ = [
    "MultiHorizonEnbPI",
    "MultiHorizonEnbPIResult",
    "Regressor",
]


@dataclass(frozen=True)
class MultiHorizonEnbPIResult:
    """Multi-horizon online run summary.

    Fields
    ------
    lower, upper, point:
        ``(T, H)`` arrays; column ``h-1`` is the band for horizon ``h``,
        issued at each test origin before its target was observed.
    marginal_coverage:
        ``(H,)`` empirical coverage per horizon over evaluable origins
        (``T - h`` origins for horizon ``h``).
    joint_coverage:
        Fraction of origins ``t`` with ``t + H`` inside the test window for
        which **all** H intervals covered ``y_{t+1..t+H}`` simultaneously.
    mean_width:
        ``(H,)`` mean interval width per horizon over all T origins.
    joint_coverage_bound:
        Bonferroni union bound ``max(0, 1 - sum_h alpha_h)`` on joint
        coverage — a worst-case bound, not a proper score of the run.
    """

    lower: Array
    upper: Array
    point: Array
    marginal_coverage: Array
    joint_coverage: float
    mean_width: Array
    joint_coverage_bound: float


class MultiHorizonEnbPI:
    """EnbPI with one residual ensemble per horizon h = 1..H.

    Parameters
    ----------
    model_factory:
        Zero-arg callable returning a fresh unfitted regressor with the
        sklearn ``fit(X, y)`` / ``predict(X)`` contract (one ensemble per
        horizon, ``n_estimators * H`` fits in total).
    horizons:
        Number of horizons H >= 1; targets are ``y_{t+1}..y_{t+H}``.
    n_estimators:
        Bootstrap models B per horizon (must be >= 2).
    alpha:
        Target miscoverage in (0, 1). Per-horizon level is ``alpha / H``
        when ``bonferroni_joint`` else ``alpha``.
    block_size:
        Circular block length for the bootstrap; 1 = i.i.d. bootstrap.
    aggregate:
        'mean' or 'median' ensemble aggregation (phi in the paper).
    seed:
        RNG seed; bootstraps are drawn horizon-major (h outer, b inner), so
        ``H = 1`` reproduces ``quant_fund.models.enbpi.EnbPI`` exactly.
    bonferroni_joint:
        Calibrate per-horizon levels at ``alpha / H`` so the union-bound
        joint coverage target is ``1 - alpha``.
    """

    def __init__(
        self,
        model_factory: Callable[[], Regressor],
        horizons: int = 1,
        n_estimators: int = 20,
        alpha: float = 0.1,
        block_size: int = 1,
        aggregate: str = "mean",
        seed: int = 0,
        bonferroni_joint: bool = False,
    ) -> None:
        if not isinstance(horizons, (int, np.integer)) or int(horizons) < 1:
            raise ValueError("horizons must be an integer >= 1")
        if n_estimators < 2:
            raise ValueError("n_estimators must be >= 2")
        a = float(alpha)
        if not 0.0 < a < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if aggregate not in ("mean", "median"):
            raise ValueError("aggregate must be 'mean' or 'median'")
        self.model_factory = model_factory
        self.horizons = int(horizons)
        self.n_estimators = int(n_estimators)
        self.alpha = a
        self.alpha_per_horizon = a / self.horizons if bonferroni_joint else a
        if not 0.0 < self.alpha_per_horizon < 1.0:
            raise ValueError("alpha / horizons must remain in (0, 1)")
        self.block_size = int(block_size)
        self.aggregate = aggregate
        self.seed = int(seed)
        self.bonferroni_joint = bool(bonferroni_joint)
        self._models: list[list[Regressor]] = []
        self._residuals: list[Array] = []
        self._window_sizes: list[int] = []
        self._n_features = 0

    # ------------------------------------------------------------------ fit
    def _agg(self, preds: Array, axis: int = 0) -> Array:
        if self.aggregate == "median":
            return np.asarray(np.median(preds, axis=axis), dtype=float)
        return np.asarray(np.mean(preds, axis=axis), dtype=float)

    def _agg_masked(self, preds: Array, mask: NDArray[np.bool_]) -> Array:
        masked = np.where(mask, preds, np.nan)
        if self.aggregate == "median":
            return np.asarray(np.nanmedian(masked, axis=0), dtype=float)
        return np.asarray(np.nanmean(masked, axis=0), dtype=float)

    def fit(self, X: Array, y: Array) -> MultiHorizonEnbPI:
        """Fit one bootstrap ensemble per horizon on the pairs (x_i, y_{i+h})."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n, p)")
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError("X and y length mismatch")
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            raise ValueError("X and y must be finite")
        min_n = max(10, int(np.ceil(1.0 / self.alpha_per_horizon)) + 1)
        if n < self.horizons + min_n:
            raise ValueError(
                f"need at least {self.horizons + min_n} training points for "
                f"H={self.horizons}, alpha_h={self.alpha_per_horizon:g}"
            )

        rng = np.random.default_rng(self.seed)
        models: list[list[Regressor]] = []
        residuals: list[Array] = []
        window_sizes: list[int] = []
        for h in range(1, self.horizons + 1):
            n_h = n - h
            X_h, y_h = X[:n_h], y[h:]
            in_bag = np.zeros((self.n_estimators, n_h), dtype=bool)
            preds = np.empty((self.n_estimators, n_h), dtype=float)
            h_models: list[Regressor] = []
            for b in range(self.n_estimators):
                idx = circular_block_bootstrap_indices(n_h, self.block_size, rng)
                in_bag[b, idx] = True
                m = self.model_factory()
                m.fit(X_h[idx], y_h[idx])
                pred = np.asarray(m.predict(X_h), dtype=float).ravel()
                if pred.shape != (n_h,) or not np.all(np.isfinite(pred)):
                    raise ValueError("base learner predict must return n finite values")
                preds[b] = pred
                h_models.append(m)
            oob = ~in_bag
            if np.any(oob.sum(axis=0) == 0):
                raise ValueError(
                    f"some training points appear in every bootstrap sample at horizon {h}; "
                    "increase n_estimators or reduce block_size"
                )
            eps = y_h - self._agg_masked(preds, oob)
            if not np.all(np.isfinite(eps)):
                raise ValueError(f"LOO residuals are not finite at horizon {h}")
            models.append(h_models)
            residuals.append(eps)
            window_sizes.append(n_h)
        self._models = models
        self._residuals = residuals
        self._window_sizes = window_sizes
        self._n_features = X.shape[1]
        return self

    def _check_fitted(self) -> None:
        if not self._models:
            raise RuntimeError("MultiHorizonEnbPI is not fitted")

    @property
    def residuals(self) -> list[Array]:
        """Per-horizon signed LOO residual windows (copies), length H."""
        self._check_fitted()
        return [r.copy() for r in self._residuals]

    @property
    def joint_coverage_bound_(self) -> float:
        """Bonferroni union bound ``max(0, 1 - H * alpha_h)`` on joint coverage."""
        return max(0.0, 1.0 - self.horizons * self.alpha_per_horizon)

    # -------------------------------------------------------------- predict
    def predict_point(self, X: Array) -> Array:
        """``(m, H)`` ensemble point forecasts, one column per horizon."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._n_features:
            raise ValueError("X must be 2-D with the training feature count")
        if X.shape[0] == 0:
            raise ValueError("X must be non-empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must be finite")
        cols = []
        for h_models in self._models:
            preds = np.stack([np.asarray(m.predict(X), dtype=float).ravel() for m in h_models])
            if preds.shape[1] != X.shape[0] or not np.all(np.isfinite(preds)):
                raise ValueError("base learner predict must return n finite values")
            cols.append(self._agg(preds, axis=0))
        return np.column_stack(cols)

    def predict_interval(self, X: Array) -> tuple[Array, Array, Array]:
        """``(lower, upper, point)``, each ``(m, H)``, from the current windows."""
        point = self.predict_point(X)
        lower = np.empty_like(point)
        upper = np.empty_like(point)
        for h in range(self.horizons):
            lo, hi = enbpi_residual_bounds(self._residuals[h], self.alpha_per_horizon)
            lower[:, h] = point[:, h] + lo
            upper[:, h] = point[:, h] + hi
        return lower, upper, point

    def predict_online(self, X: Array, y: Array) -> MultiHorizonEnbPIResult:
        """Sequential multi-horizon Algorithm 1 with h-step-delayed label reveal.

        At every test origin ``t``: issue all H intervals from the current
        windows, then observe ``y_t`` and slide each horizon ``h``'s window
        with the now-revealed residual ``y_t - yhat_{t-h}^{h}`` (the forecast
        issued h steps earlier). No prediction ever sees its own target
        label. Requires ``T >= H + 1`` test points so joint coverage has at
        least one evaluable origin.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[1] != self._n_features:
            raise ValueError("X must be 2-D with the training feature count")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch")
        t_steps = X.shape[0]
        if t_steps == 0:
            raise ValueError("no test points")
        if t_steps < self.horizons + 1:
            raise ValueError(
                f"test window must have at least H + 1 = {self.horizons + 1} points "
                "to evaluate joint coverage"
            )
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            raise ValueError("X and y must be finite")

        h = self.horizons
        point = self.predict_point(X)
        lower = np.empty((t_steps, h))
        upper = np.empty((t_steps, h))
        covered = np.full((t_steps, h), np.nan)
        for j in range(t_steps):
            for k in range(h):
                lo, hi = enbpi_residual_bounds(self._residuals[k], self.alpha_per_horizon)
                lower[j, k] = point[j, k] + lo
                upper[j, k] = point[j, k] + hi
            for k in range(h):
                jp = j - (k + 1)
                if jp >= 0:
                    resid = y[j] - point[jp, k]
                    window = np.concatenate([self._residuals[k], [resid]])
                    self._residuals[k] = window[-self._window_sizes[k] :]
                    covered[jp, k] = float(lower[jp, k] <= y[j] <= upper[jp, k])
        marginal = np.array([float(np.nanmean(covered[:, k])) for k in range(h)], dtype=float)
        joint_mask = ~np.any(np.isnan(covered), axis=1)
        joint = float(np.mean(np.all(covered[joint_mask] == 1.0, axis=1)))
        return MultiHorizonEnbPIResult(
            lower=lower,
            upper=upper,
            point=point,
            marginal_coverage=marginal,
            joint_coverage=joint,
            mean_width=np.asarray(np.mean(upper - lower, axis=0), dtype=float),
            joint_coverage_bound=self.joint_coverage_bound_,
        )
