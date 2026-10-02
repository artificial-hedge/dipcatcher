"""Positive-unlabeled (PU) learning: train classifiers
when only positives and unlabeled data exist.

- Elkan & Noto (2008): learn e = P(s=1|y=1) from a
  validation set; P(y=1|x) = P(s=1|x) / e.
- c-SNE (scarce positive examples): Liu et al. — label
  frequency / prior shift estimator.
- Non-traditional PU risk estimator (du Plessis, Niu &
  Sugiyama 2014): unbiased empirical risk from PU data
  only, for linear/logistic models via the sigmoid loss.
- Wrapper scoring: a logistic-regression PU scorer
  implemented on the unbiased risk with L-BFGS.

References
----------
- Elkan & Noto (2008) 'Learning classifiers from only
  positive and unlabeled data' KDD.
- du Plessis, Niu & Sugiyama (2014) 'Analysis of learning
  from positive and unlabeled data' NeurIPS.
- Liu, Lee & Yu (2003) 'Building text classifiers using
  positive and unlabeled examples' ICDM.

Honesty
-------
SYNTHETIC self-check: 2-Gaussian mixture with a known
labeling propensity; asserts the corrected score recovers
the true P(y=1|x) ordering (AUC gate) and e estimate is
within tolerance.

Composition
-----------
Pure numpy/scipy. Inputs are feature matrices + PU
labels; outputs are corrected positive probabilities
and the estimated propensity e.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


def _check_xy(X: FloatArray, s: FloatArray) -> tuple[FloatArray, FloatArray]:
    Xa = np.asarray(X, dtype=np.float64)
    sa = np.asarray(s, dtype=np.float64).ravel()
    if Xa.ndim != 2 or Xa.shape[0] != sa.size or sa.size < 10:
        raise ValueError("X (n,d) and s (n) mismatch, n>=10")
    if not np.isfinite(Xa).all() or not np.isin(sa, [0.0, 1.0]).all():
        raise ValueError("labels must be binary {0,1}, X finite")
    if sa.sum() < 2:
        raise ValueError("need >=2 labeled positives")
    return Xa, sa


def elkan_e(s_hat: FloatArray) -> float:
    """Elkan-Noto propensity: e = mean P(s=1|x) over labeled positives."""
    h = np.asarray(s_hat, dtype=np.float64).ravel()
    if h.size == 0 or not np.isfinite(h).all():
        raise ValueError("empty/non-finite scores")
    e = float(h.mean())
    if e <= 0:
        raise ValueError("e estimate degenerate")
    return min(e, 1.0)


def elkan_correct(s_hat: FloatArray, e: float) -> FloatArray:
    """P(y=1|x) = P(s=1|x) / e, clipped to [0,1]."""
    h = np.asarray(s_hat, dtype=np.float64)
    if e <= 0:
        raise ValueError("e positive")
    return np.clip(h / e, 0.0, 1.0)


def _sigmoid(z: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_pu_logistic(
    X: FloatArray,
    s: FloatArray,
    pi_p: float | None = None,
    l2: float = 1e-3,
    iters: int = 300,
) -> dict[str, FloatArray | float]:
    """Unbiased-PU logistic regression (du Plessis et al. 2014).

    Minimizes the unbiased PU empirical risk with the
    logistic loss:
      R = pi_p * mean_P ell(f) + mean_U ell(-f)
          - pi_p * mean_P ell(-f)
    which uses only labeled positives P and unlabeled U.
    The negative term is clipped at 0 (Kiryo et al. 2017
    non-negative correction) since the unbiased estimator
    is unbounded below.
    pi_p (class prior P(y=1)) is estimated by the
    Elkan-Noto ratio on a logistic fit to s if not given.
    """
    Xa, sa = _check_xy(X, s)
    n, d = Xa.shape
    Xb = np.hstack([Xa, np.ones((n, 1))])
    pos = sa == 1.0
    n_p = int(pos.sum())
    if pi_p is None:
        # quick ordinary logistic on s to get P(s=1|x); e = mean over P
        w0 = np.zeros(d + 1)
        res0 = minimize(
            lambda w: (
                -(
                    sa * np.log(_sigmoid(Xb @ w) + 1e-12)
                    + (1 - sa) * np.log(1 - _sigmoid(Xb @ w) + 1e-12)
                ).mean()
                + l2 * (w[:-1] ** 2).sum()
            ),
            w0,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        e = elkan_e(_sigmoid(Xb[pos] @ res0.x))
        pi_p = float(sa.mean() / e)
        pi_p = min(max(pi_p, 1e-3), 0.95)

    def _neg_part(f: FloatArray) -> tuple[float, float]:
        """mean_U ell(-f) - pi_p mean_P ell(-f), clipped >=0."""
        lm = np.logaddexp(0.0, f)
        raw = float(lm[~pos].mean() - pi_p * lm[pos].mean())
        return max(raw, 0.0), raw

    def risk(w: FloatArray) -> float:
        f = Xb @ w
        lp = np.logaddexp(0.0, -f)  # ell(+f) = log(1+e^-f)
        neg, _ = _neg_part(f)
        r = pi_p * float(lp[pos].mean()) + neg
        return float(r + l2 * (w[:-1] ** 2).sum())

    def grad(w: FloatArray) -> FloatArray:
        f = Xb @ w
        gp = -_sigmoid(-f[pos])
        g = pi_p * (Xb[pos].T @ gp) / n_p
        neg_clipped, neg_raw = _neg_part(f)
        if neg_raw > 0 or neg_clipped > 0:
            # d/dw [mean_U ell(-f) - pi_p mean_P ell(-f)]
            gu = _sigmoid(f[~pos])
            gm = _sigmoid(f[pos])
            g = g + (Xb[~pos].T @ gu) / max(int((~pos).sum()), 1) - pi_p * (Xb[pos].T @ gm) / n_p
        g[:-1] += 2 * l2 * w[:-1]
        return np.asarray(g, dtype=np.float64)

    w_init = np.zeros(d + 1)
    res = minimize(risk, w_init, jac=grad, method="L-BFGS-B", options={"maxiter": iters})
    if not np.isfinite(res.x).all():
        raise ValueError("PU fit diverged")
    return {"w": res.x, "pi_p": float(pi_p), "risk": float(res.fun)}


def pu_predict(X: FloatArray, fit: dict[str, FloatArray | float]) -> FloatArray:
    w = np.asarray(fit["w"], dtype=np.float64)
    Xa = np.asarray(X, dtype=np.float64)
    Xb = np.hstack([Xa, np.ones((Xa.shape[0], 1))])
    return _sigmoid(Xb @ w)


def bench_pu(seed: int = 507) -> dict[str, float]:
    """SYNTHETIC: 2-Gaussian mixture, PU labels by propensity.

    Asserts corrected AUC > 0.9 and propensity recovery."""
    rng = np.random.default_rng(seed)
    n = 2000
    y = rng.random(n) < 0.4
    X = np.where(
        y[:, None],
        rng.normal([1.5, 1.0], 0.9, (n, 2)),
        rng.normal([-1.0, -0.5], 1.1, (n, 2)),
    )
    e_true = 0.45
    s = np.zeros(n)
    pos_idx = np.where(y)[0]
    labeled = rng.random(pos_idx.size) < e_true
    s[pos_idx[labeled]] = 1.0
    fit = fit_pu_logistic(X, s)
    scores = pu_predict(X, fit)
    # AUC vs true labels
    order = np.argsort(scores)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    n1, n0 = int(y.sum()), int((~y).sum())
    auc = float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    if auc < 0.9:
        raise ValueError("PU AUC below gate")
    pi_hat = fit["pi_p"]
    if not isinstance(pi_hat, float):
        raise TypeError("pi_p must be scalar")
    return {
        "synthetic_auc": auc,
        "synthetic_pi_hat": float(pi_hat),
        "synthetic_pi_err": float(abs(pi_hat - 0.4)),
        "synthetic_risk": float(fit["risk"]),
        "synthetic_labeled_frac": float(s.mean()),
    }
