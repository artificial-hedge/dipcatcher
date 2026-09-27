"""NGBoost-style natural-gradient distributional boosting (lite).

Natural-gradient boosting for probabilistic regression (Duan, Avati, Ding,
Thai, Basu, Ng, Schuler, 2020, ICML, PMLR 119:2690-2700, "NGBoost: Natural
Gradient Boosting for Probabilistic Prediction", arXiv:1910.03225). A
boosting loop over a two-parameter location-scale family: each round computes
the per-sample gradient of a proper scoring rule (Gneiting & Raftery, 2007,
JASA 102:359-378) w.r.t. the distribution parameters (mu, log_sigma),
rescales it by the inverse Fisher information (the "natural gradient" of
Amari, 1998, "Natural Gradient Works Efficiently in Learning", NeurIPS 11),
and fits one small regression tree per parameter to the negative natural
gradient, stepping parameters by the learning rate.

Distribution and scoring-rule support:

- ``dist='normal'``: per-sample parameters (mu, log_sigma). CRPS gradients
  are closed-form, differentiating the Gaussian CRPS of Gneiting, Balabdaoui
  & Raftery, 2007 (the closed form appears in Gneiting et al., 2005,
  "Calibrated Probabilistic Forecasting Using Ensemble Model Output
  Statistics and Minimum CRPS Estimation", MWR 133:1098-1118, eq. (15)):
  with z = (y - mu)/sigma,

      CRPS = sigma * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ]
      d CRPS/d mu      = 1 - 2 Phi(z)          (dz/dmu = -1/sigma cancels)
      d CRPS/d log_sig = sigma (2 phi(z) - 1/sqrt(pi))

  using d/dz [z(2Phi(z)-1) + 2phi(z)] = 2Phi(z) - 1 (the z*phi terms cancel).
- ``dist='student_t'``: (mu, log_sigma) with the degrees of freedom FIXED at
  construction (default df=5). Fixing df is not merely computational: for
  the location-scale t with free (sigma, nu), the Fisher information block
  for (sigma, nu) has NONZERO off-diagonal terms (mu is orthogonal to both
  by symmetry, but scale and shape are not — see standard t-information
  results tabulated e.g. in Lange, Little & Taylor, 1989, JASA 84:881-896),
  so a diagonal/closed-form natural gradient would be invalid if nu were
  fit. With nu fixed, the Fisher matrix for (mu, log_sigma) is exactly
  diagonal with closed form

      I_mumu     = (nu + 1) / ((nu + 3) sigma^2)
      I_loglog   = 2 nu / (nu + 3)

  (scale-by-sigma^2 of the sigma-sigma entry 2 nu/((nu+3) sigma^2); the
  normal limits nu -> inf recover (1/sigma^2, 2), as they must).
  The Student-t CRPS itself has a closed form (Jordan, Krüger & Lerch,
  2019, "Evaluating Probabilistic Forecasts with scoringRules", JSS
  90:1-37; implemented in quant_fund.metrics.scoring.crps_student_t), so no
  numerical CRPS fallback is needed; only the GRADIENTS of that closed form
  are obtained by central differences (step h = 1e-4 * max(1, |.|),
  documented; truncation error O(h^2), far inside tree-fitting tolerance).
- ``score='mle'``: the log-score gradients of the negative log-likelihood,
  natural-gradient scaled by the same diagonal Fisher inverses. For the
  normal this collapses to the classic boosting targets (y - mu) and
  (z^2 - 1)/2; for the t they are (nu+3) sigma z / (nu + z^2) and
  (nu+3)/(2 nu) * ((nu+1) z^2/(nu+z^2) - 1).

Honesty contract: only proper scores (CRPS, log-score) are exposed. No
Sharpe/Sortino/P&L content anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr
from sklearn.tree import DecisionTreeRegressor

from quant_fund.metrics.scoring import crps_gaussian, crps_student_t

Array = NDArray[np.float64]

__all__ = ["NGBoostLite"]

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)
_LOG_SIGMA_CLIP = 10.0  # |log_sigma| <= 10  =>  sigma in [4.5e-5, 2.2e4]


@dataclass(frozen=True)
class _FisherDiag:
    """Per-sample diagonal Fisher entries and their inverses for (mu, log_sigma).

    ``inv_mu`` = I_mumu^{-1} multiplies the mu-gradient; ``inv_log`` =
    I_loglog^{-1} multiplies the log_sigma-gradient. Both are functions of
    sigma only (df fixed), which keeps the natural gradient closed-form.
    """

    inv_mu: Array
    inv_log: Array


def _fisher_diag(dist: str, sigma: Array, df: float) -> _FisherDiag:
    """Inverse diagonal Fisher information for (mu, log_sigma); see module docstring."""
    s2 = sigma * sigma
    if dist == "normal":
        inv_mu = s2
        inv_log = np.full_like(sigma, 0.5)
    else:  # student_t, df fixed (validated df > 2 at construction)
        inv_mu = (df + 3.0) * s2 / (df + 1.0)
        inv_log = np.full_like(sigma, (df + 3.0) / (2.0 * df))
    return _FisherDiag(inv_mu=inv_mu, inv_log=inv_log)


def _normal_crps_grad(y: Array, mu: Array, sigma: Array) -> tuple[Array, Array]:
    """Closed-form d CRPS/d mu and d CRPS/d log_sigma for N(mu, sigma^2).

    Derivations in the module docstring (Gneiting et al. 2005, MWR, eq. (15)
    and its z-derivative; the 2 z phi(z) terms cancel exactly).
    """
    z = (y - mu) / sigma
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    d_mu = 1.0 - 2.0 * ndtr(z)
    d_log = sigma * (2.0 * phi - _INV_SQRT_PI)
    return d_mu, d_log


def _student_t_crps_grad(y: Array, mu: Array, log_sigma: Array, df: float) -> tuple[Array, Array]:
    """d CRPS/d mu and d CRPS/d log_sigma by central differences.

    The Student-t CRPS closed form (Jordan, Krüger & Lerch 2019, JSS;
    quant_fund.metrics.scoring.crps_student_t) has no tidy elementary
    gradient, so central differences on that closed form are used:
    g(p) ~= (S(p + h) - S(p - h)) / (2 h), h = 1e-4 * max(1, |p|). Truncation
    error is O(h^2); accuracy is far inside the tolerance a depth-2 tree fit
    can exploit.
    """
    h_mu = 1e-4 * np.maximum(1.0, np.abs(mu))
    h_ls = 1e-4 * np.maximum(1.0, np.abs(log_sigma))

    def _score(mu_v: Array, ls_v: Array) -> Array:
        return crps_student_t(y, mu_v, np.exp(ls_v), df)

    d_mu = (_score(mu + h_mu, log_sigma) - _score(mu - h_mu, log_sigma)) / (2.0 * h_mu)
    d_log = (_score(mu, log_sigma + h_ls) - _score(mu, log_sigma - h_ls)) / (2.0 * h_ls)
    return np.asarray(d_mu, dtype=float), np.asarray(d_log, dtype=float)


def _normal_mle_natural(y: Array, mu: Array, sigma: Array) -> tuple[Array, Array]:
    """Negative natural gradient of the normal NLL: (y - mu), (z^2 - 1)/2.

    d NLL/d log_sigma = 1 - z^2 (NLL = z^2/2 + log_sigma + const) and the
    Fisher inverse for log_sigma is 1/2, so the negative natural gradient is
    (z^2 - 1)/2 — positive when sigma is too small for the residual, pushing
    log_sigma up, as it must.
    """
    z = (y - mu) / sigma
    return y - mu, 0.5 * (z * z - 1.0)


def _student_t_mle_natural(y: Array, mu: Array, sigma: Array, df: float) -> tuple[Array, Array]:
    """Negative natural gradient of the t NLL; see module docstring."""
    z = (y - mu) / sigma
    denom = df + z * z
    nat_mu = (df + 3.0) * sigma * z / denom
    nat_log = (df + 3.0) / (2.0 * df) * ((df + 1.0) * z * z / denom - 1.0)
    return nat_mu, nat_log


class NGBoostLite:
    """Natural-gradient boosting over (mu, log_sigma) of a location-scale family.

    Parameters
    ----------
    dist:
        'normal' or 'student_t' (anything else raises ValueError).
    score:
        'crps' (Gneiting-Raftery proper score) or 'mle' (log-score).
    n_estimators, learning_rate, max_depth, min_samples_leaf:
        Boosting rounds and base-learner controls; each round fits one
        ``DecisionTreeRegressor`` per parameter to the negative natural
        gradient and steps parameters by ``learning_rate``.
    random_state:
        Seeds base learners and the sampling generator; None is allowed.
    df:
        Fixed degrees of freedom for ``dist='student_t'`` (must exceed 2 so
        the variance, the Fisher entries, and the CRPS closed form exist).
        Fixing df keeps the Fisher matrix for (mu, log_sigma) diagonal and
        closed-form (module docstring).

    Fail-closed: degenerate inputs (n < 50, n < 10*max(n_features, 2),
    constant ``y``, NaN/inf in ``X``/``y``) raise ValueError at ``fit``;
    any predict/score method before ``fit`` raises RuntimeError.
    """

    def __init__(
        self,
        dist: str = "normal",
        score: str = "crps",
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 2,
        min_samples_leaf: int = 10,
        random_state: int | None = None,
        df: float = 5.0,
    ) -> None:
        if dist not in ("normal", "student_t"):
            raise ValueError("dist must be 'normal' or 'student_t'")
        if score not in ("crps", "mle"):
            raise ValueError("score must be 'crps' or 'mle'")
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be >= 1")
        if not np.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if int(max_depth) < 1:
            raise ValueError("max_depth must be >= 1")
        if int(min_samples_leaf) < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if dist == "student_t" and (not np.isfinite(float(df)) or float(df) <= 2.0):
            raise ValueError("df must be finite and > 2 (finite variance required)")
        self.dist = dist
        self.score = score
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state
        self.df = float(df)
        self._fitted = False
        self._base_mu = 0.0
        self._base_log_sigma = 0.0
        self._trees_mu: list[DecisionTreeRegressor] = []
        self._trees_log: list[DecisionTreeRegressor] = []

    # ------------------------------------------------------------------ #
    # fitting                                                            #
    # ------------------------------------------------------------------ #

    def _validate_fit_data(self, X: Array, y: Array) -> tuple[Array, Array]:
        x = np.asarray(X, dtype=float)
        yv = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2:
            raise ValueError("X must be a 2-D (n, n_features) array")
        n, n_feat = int(x.shape[0]), int(x.shape[1])
        if yv.shape[0] != n:
            raise ValueError("y must have one entry per row of X")
        if not np.all(np.isfinite(x)):
            raise ValueError("X must contain only finite values (NaN/inf rejected)")
        if not np.all(np.isfinite(yv)):
            raise ValueError("y must contain only finite values (NaN/inf rejected)")
        if n < 50:
            raise ValueError(f"need at least 50 samples, got {n}")
        if n < 10 * max(n_feat, 2):
            raise ValueError(f"need n >= 10*max(n_features, 2) = {10 * max(n_feat, 2)}, got {n}")
        if float(np.ptp(yv)) == 0.0:
            raise ValueError("constant y is degenerate for distributional fitting")
        return x, yv

    def _negative_natural_gradient(
        self, y: Array, mu: Array, log_sigma: Array
    ) -> tuple[Array, Array]:
        """Tree targets: -F^{-1} grad S, i.e. the natural-gradient DESCENT direction."""
        sigma = np.exp(log_sigma)
        if self.score == "mle":
            if self.dist == "normal":
                return _normal_mle_natural(y, mu, sigma)
            return _student_t_mle_natural(y, mu, sigma, self.df)
        if self.dist == "normal":
            g_mu, g_log = _normal_crps_grad(y, mu, sigma)
        else:
            g_mu, g_log = _student_t_crps_grad(y, mu, log_sigma, self.df)
        fisher = _fisher_diag(self.dist, sigma, self.df)
        return -fisher.inv_mu * g_mu, -fisher.inv_log * g_log

    def fit(self, X: Array, y: Array) -> NGBoostLite:
        """Fit the boosting loop; returns ``self``.

        Parameters are initialized at the global MLE (mu = mean(y),
        log_sigma = log std(y)) and refined by ``n_estimators`` rounds of
        natural-gradient tree boosting.
        """
        x, yv = self._validate_fit_data(X, y)
        self._base_mu = float(np.mean(yv))
        self._base_log_sigma = float(np.log(np.std(yv)))
        mu = np.full(x.shape[0], self._base_mu, dtype=float)
        log_sigma = np.full(x.shape[0], self._base_log_sigma, dtype=float)
        self._trees_mu = []
        self._trees_log = []
        for _ in range(self.n_estimators):
            t_mu, t_log = self._negative_natural_gradient(yv, mu, log_sigma)
            tree_mu = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            tree_log = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            tree_mu.fit(x, t_mu)
            tree_log.fit(x, t_log)
            mu = mu + self.learning_rate * tree_mu.predict(x)
            log_sigma = np.clip(
                log_sigma + self.learning_rate * tree_log.predict(x),
                -_LOG_SIGMA_CLIP,
                _LOG_SIGMA_CLIP,
            )
            self._trees_mu.append(tree_mu)
            self._trees_log.append(tree_log)
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # prediction                                                         #
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("NGBoostLite is not fitted yet; call fit(X, y) first")

    def _validate_X(self, X: Array) -> Array:
        x = np.asarray(X, dtype=float)
        if x.ndim != 2:
            raise ValueError("X must be a 2-D (n, n_features) array")
        if not np.all(np.isfinite(x)):
            raise ValueError("X must contain only finite values (NaN/inf rejected)")
        return x

    def predict_params(self, X: Array) -> tuple[Array, Array]:
        """Per-sample (mu, sigma); sigma = exp(log_sigma) > 0 always."""
        self._check_fitted()
        x = self._validate_X(X)
        mu = np.full(x.shape[0], self._base_mu, dtype=float)
        log_sigma = np.full(x.shape[0], self._base_log_sigma, dtype=float)
        for tree_mu, tree_log in zip(self._trees_mu, self._trees_log, strict=True):
            mu = mu + self.learning_rate * tree_mu.predict(x)
            log_sigma = log_sigma + self.learning_rate * tree_log.predict(x)
        # Same clip as the training loop: keeps sigma in [exp(-10), exp(10)]
        # and exp() away from under/overflow.
        return mu, np.exp(np.clip(log_sigma, -_LOG_SIGMA_CLIP, _LOG_SIGMA_CLIP))

    def sample(self, X: Array, n_samples: int, rng: np.random.Generator | None = None) -> Array:
        """Draw ``n_samples`` i.i.d. predictive draws per row of ``X``; shape (n, n_samples).

        ``rng`` overrides the generator seeded from ``random_state``; pass
        your own ``np.random.default_rng(k)`` for independent streams.
        """
        self._check_fitted()
        if int(n_samples) < 1:
            raise ValueError("n_samples must be >= 1")
        mu, sigma = self.predict_params(X)
        gen = rng if rng is not None else np.random.default_rng(self.random_state)
        if self.dist == "student_t":
            draws = gen.standard_t(self.df, size=(mu.shape[0], int(n_samples)))
        else:
            draws = gen.standard_normal((mu.shape[0], int(n_samples)))
        return mu[:, None] + sigma[:, None] * draws

    def mean_crps(self, X: Array, y: Array) -> float:
        """Mean proper CRPS of the fitted predictive on (X, y).

        Gaussian closed form (Gneiting et al. 2005) or Student-t closed form
        (Jordan, Krüger & Lerch 2019), both via
        quant_fund.metrics.scoring. Lower is better; this is a research
        diagnostic, never a live-capital claim.
        """
        self._check_fitted()
        yv = np.asarray(y, dtype=float).reshape(-1)
        mu, sigma = self.predict_params(X)
        if yv.shape[0] != mu.shape[0]:
            raise ValueError("y must have one entry per row of X")
        if self.dist == "normal":
            scores = crps_gaussian(yv, mu, sigma)
        else:
            scores = crps_student_t(yv, mu, sigma, self.df)
        if not bool(np.all(np.isfinite(scores))):
            raise ValueError("non-finite CRPS encountered; inputs must be finite")
        return float(np.mean(scores))
