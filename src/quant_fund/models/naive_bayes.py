"""Naive Bayes classifiers: Gaussian NB (per-class (SYNTHETIC)
univariate Gaussians), multinomial NB on counts
(m-class Bayes with Laplace smoothing), and Bernoulli NB
for binary features. Synthetic bench gates separation
across all three data regimes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def gaussian_nb_fit(x: FloatArray, y: FloatArray) -> dict[str, object]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    mu = np.stack([x[y == c].mean(axis=0) for c in classes])
    var = np.stack([x[y == c].var(axis=0) for c in classes]) + 1e-9
    prior = np.array([np.mean(y == c) for c in classes])
    return {"classes": classes, "mu": mu, "var": var, "prior": prior}


def gaussian_nb_predict(model: dict[str, object], x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(model["mu"])
    var = np.asarray(model["var"])
    prior = np.asarray(model["prior"])
    classes = np.asarray(model["classes"])
    logp = -0.5 * np.log(2 * np.pi * var) - (x[:, None, :] - mu[None, :, :]) ** 2 / (
        2 * var[None, :, :]
    )
    scores = logp.sum(axis=2) + np.log(prior[None, :])
    return np.asarray(classes[np.argmax(scores, axis=1)])


def multinomial_nb_fit(x: FloatArray, y: FloatArray, alpha: float = 1.0) -> dict[str, object]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    theta = np.stack(
        [(x[y == c].sum(axis=0) + alpha) / (x[y == c].sum() + alpha * x.shape[1]) for c in classes]
    )
    prior = np.array([np.mean(y == c) for c in classes])
    return {"classes": classes, "theta": theta, "prior": prior}


def multinomial_nb_predict(model: dict[str, object], x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    theta = np.asarray(model["theta"])
    prior = np.asarray(model["prior"])
    classes = np.asarray(model["classes"])
    scores = x @ np.log(theta).T + np.log(prior)[None, :]
    return np.asarray(classes[np.argmax(scores, axis=1)])


def bernoulli_nb_fit(x: FloatArray, y: FloatArray, alpha: float = 1.0) -> dict[str, object]:
    x = (np.asarray(x, dtype=np.float64) > 0).astype(np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    theta = np.stack(
        [(x[y == c].sum(axis=0) + alpha) / ((y == c).sum() + 2 * alpha) for c in classes]
    )
    prior = np.array([np.mean(y == c) for c in classes])
    return {"classes": classes, "theta": theta, "prior": prior}


def bernoulli_nb_predict(model: dict[str, object], x: FloatArray) -> FloatArray:
    x = (np.asarray(x, dtype=np.float64) > 0).astype(np.float64)
    theta = np.asarray(model["theta"])
    prior = np.asarray(model["prior"])
    classes = np.asarray(model["classes"])
    scores = x @ np.log(theta).T + (1 - x) @ np.log(1 - theta).T + np.log(prior)[None, :]
    return np.asarray(classes[np.argmax(scores, axis=1)])


def bench_naive_bayes(seed: int = 564) -> dict[str, float]:
    """SYNTHETIC: Gaussian NB on well-separated blobs
    (independent features — NB is nearly optimal here),
    multinomial NB on topic-word counts, Bernoulli NB on
    binary feature presence."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n = 300
    perm = rng.permutation(n)
    tr, te = perm[:200], perm[200:]
    # Gaussian blobs
    xg = np.vstack(
        [rng.normal([0, 0, 0], 0.5, (n // 2, 3)), rng.normal([2.5, 2.5, 0], 0.5, (n // 2, 3))]
    )
    yg = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    m = gaussian_nb_fit(xg[tr], yg[tr])
    out["synthetic_gnb_acc"] = float((gaussian_nb_predict(m, xg[te]) == yg[te]).mean())
    if out["synthetic_gnb_acc"] < 0.95:
        raise ValueError(f"gnb acc off: {out['synthetic_gnb_acc']}")
    # multinomial: class 0 docs weight words 0-7, class 1 words 8-15
    v = 16
    docs = []
    labels = []
    for i in range(200):
        p = np.full(v, 0.02)
        p[:8] += 0.9 if i < 100 else 0.0
        p[8:] += 0.0 if i < 100 else 0.9
        p /= p.sum()
        docs.append(rng.multinomial(40, p))
        labels.append(i < 100)
    xm = np.asarray(docs, dtype=np.float64)
    ym = np.asarray(labels, dtype=np.float64)
    perm2 = rng.permutation(200)
    mm = multinomial_nb_fit(xm[perm2[:140]], ym[perm2[:140]])
    out["synthetic_mnb_acc"] = float(
        (multinomial_nb_predict(mm, xm[perm2[140:]]) == ym[perm2[140:]]).mean()
    )
    if out["synthetic_mnb_acc"] < 0.9:
        raise ValueError(f"mnb acc off: {out['synthetic_mnb_acc']}")
    # Bernoulli: class 1 has features 0-9 on
    xb = (rng.uniform(0, 1, (n, 20)) < np.where(np.arange(20)[None, :] < 10, 0.8, 0.1)).astype(
        float
    )
    xb[n // 2 :] = 1 - xb[n // 2 :]
    yb = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    bm = bernoulli_nb_fit(xb[tr], yb[tr])
    out["synthetic_bnb_acc"] = float((bernoulli_nb_predict(bm, xb[te]) == yb[te]).mean())
    if out["synthetic_bnb_acc"] < 0.9:
        raise ValueError(f"bnb acc off: {out['synthetic_bnb_acc']}")
    return out
