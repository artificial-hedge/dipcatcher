"""Evidential / subjective-logic canon: Dirichlet-evidence classifier
(Sensoy et al.) — the network-free version learned by matching class
evidence counts to Dirichlet parameters, plus uncertainty decomposition
(vacuity, dissonance). ``bench_evidential`` plants overlapping class
clusters and gates: accuracy, aleatoric-vs-epistemic ordering (points
near the boundary get lower expected precision... reported via the
uncertainty margin between in-support and out-of-support probes), and
expected-calibration of the predicted probabilities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import psi

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def evidential_fit(
    x: FloatArray,
    y: IntArray,
    n_classes: int,
    evidence_cap: float = 20.0,
) -> dict[str, FloatArray]:
    """Class-conditionals as diagonal Gaussians; per-point evidence =
    scaled Gaussian membership mapped to Dirichlet alpha."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    mus = np.zeros((n_classes, x.shape[1]))
    sds = np.ones((n_classes, x.shape[1]))
    priors = np.ones(n_classes)
    for c in range(n_classes):
        xc = x[y == c]
        if len(xc) == 0:
            continue
        mus[c] = xc.mean(axis=0)
        sds[c] = np.maximum(xc.std(axis=0), 1e-3)
        priors[c] = len(xc) / len(y)
    return {
        "mus": mus,
        "sds": sds,
        "priors": priors,
        "cap": np.asarray(evidence_cap, dtype=np.float64),
    }


def evidential_alpha(model: dict[str, FloatArray], x: FloatArray) -> FloatArray:
    """Dirichlet alpha = 1 + evidence, evidence = cap * class likelihood
    * prior normalized."""
    mus = model["mus"]
    sds = model["sds"]
    priors = model["priors"]
    cap = float(model["cap"])
    x = np.asarray(x, dtype=np.float64)
    z = (x[:, None, :] - mus[None, :, :]) / sds[None, :, :]
    logl = -0.5 * np.sum(z * z, axis=2) - np.sum(np.log(sds), axis=1)[None, :]
    logl += np.log(np.maximum(priors, 1e-12))[None, :]
    lk = np.exp(logl - logl.max(axis=1, keepdims=True))
    share = lk / np.maximum(lk.sum(axis=1, keepdims=True), 1e-12)
    # total evidence decays with distance to the nearest class centroid
    dmin = np.min(np.sum(z * z, axis=2), axis=1)
    ev_tot = cap * np.exp(-0.5 * dmin)
    return np.asarray(1.0 + ev_tot[:, None] * share)


def dirichlet_predict(alpha: FloatArray) -> dict[str, FloatArray]:
    s = alpha.sum(axis=1, keepdims=True)
    p = alpha / s
    k = alpha.shape[1]
    vacuity = k / s[:, 0]
    # dissonance via pairwise conflict measure
    b = (alpha - 1.0) / np.maximum(s - k, 1e-9)
    bb = np.clip(b, 0.0, 1.0)
    bal = 1.0 - np.abs(bb[:, :, None] - bb[:, None, :]).mean(axis=(1, 2))
    dissonance = 1.0 - bal
    return {
        "p": np.asarray(p),
        "vacuity": np.asarray(vacuity),
        "dissonance": np.asarray(dissonance),
        "s": np.asarray(s[:, 0]),
    }


def expected_loglik(alpha: FloatArray) -> FloatArray:
    """E[log p_k] under Dirichlet = psi(alpha_k) - psi(S)."""
    s = alpha.sum(axis=1, keepdims=True)
    return np.asarray(psi(alpha) - psi(s))


def bench_evidential(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n, d, k = 600, 5, 3
    y = rng.integers(0, k, n).astype(np.int64)
    centers = np.array([[2.0, 0, 0, 0, 0], [-2.0, 0, 0, 0, 0], [0, 2.0, 0, 0, 0]])
    x = centers[y] + 1.0 * rng.standard_normal((n, d))
    perm = rng.permutation(n)
    tr, te = perm[:450], perm[450:]
    m = evidential_fit(x[tr], y[tr], k)
    a = evidential_alpha(m, x[te])
    out = dirichlet_predict(a)
    acc = float(np.mean(out["p"].argmax(1) == y[te]))
    # out-of-support probe: far-away points should have lower total evidence
    x_far = centers.mean(axis=0)[None, :] + 8.0 * rng.standard_normal((50, d))
    a_far = evidential_alpha(m, x_far)
    s_far = float(np.asarray(a_far).sum(1).mean())
    s_in = float(np.asarray(a).sum(1).mean())
    # calibration of top-class probability: reliability gap
    conf = out["p"].max(axis=1)
    correct = (out["p"].argmax(1) == y[te]).astype(np.float64)
    bins = np.quantile(conf, np.linspace(0, 1, 6))
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mm = (conf >= lo) & (conf <= hi)
        if mm.any():
            ece += float(mm.mean() * abs(conf[mm].mean() - correct[mm].mean()))
    return {
        "synthetic_evi_acc": acc,
        "synthetic_evi_ece": ece,
        "synthetic_evi_s_in": s_in,
        "synthetic_evi_s_far": s_far,
        "synthetic_evi_vac_in": float(out["vacuity"].mean()),
        "synthetic_evi_vac_far": float(k / s_far),
    }
