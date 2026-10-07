"""Fisher discriminant analysis: LDA on the pooled within- (SYNTHETIC)
class covariance, QDA on per-class covariances with shrinkage
(regularized DA, Friedman 1989). Synthetic bench gates
that QDA dominates LDA when class covariances differ."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def lda_fit(x: FloatArray, y: FloatArray, shrink: float = 0.05) -> dict[str, object]:
    """Pooled within-class Σ; predict via argmax of Fisher
    linear discriminants."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    mu = [x[y == c].mean(axis=0) for c in classes]
    cov = np.zeros((x.shape[1], x.shape[1]))
    for c, m in zip(classes, mu, strict=True):
        d = x[y == c] - m
        cov += d.T @ d
    cov /= max(len(y) - len(classes), 1)
    cov = (1 - shrink) * cov + shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    return {"classes": classes, "mu": mu, "cov": cov}


def _log_gauss(x: FloatArray, mu: FloatArray, cov: FloatArray) -> FloatArray:
    """log N(x; mu, Σ) up to the common constant."""
    prec = np.linalg.inv(cov + 1e-8 * np.eye(cov.shape[0]))
    d = x - mu
    sign, logdet = np.linalg.slogdet(cov + 1e-8 * np.eye(cov.shape[0]))
    quad = np.einsum("ij,jk,ik->i", d, prec, d)
    return np.asarray(-0.5 * (quad + logdet))


def lda_predict(model: dict[str, object], x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    classes = np.asarray(model["classes"])
    mu = model["mu"]
    if not (isinstance(mu, list)):
        raise ValueError("isinstance(mu, list)")
    cov = np.asarray(model["cov"])
    scores = np.stack(
        [_log_gauss(x, np.asarray(m), cov) for m in mu],
        axis=1,
    )
    return np.asarray(classes[np.argmax(scores, axis=1)])


def qda_fit(
    x: FloatArray, y: FloatArray, shrink: float = 0.1, pooled: float = 0.0
) -> dict[str, object]:
    """Per-class covariances, shrunk toward the pooled Σ then
    toward the diagonal."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    mu = [x[y == c].mean(axis=0) for c in classes]
    cov_pooled = np.cov(x.T)
    covs = []
    for c, m in zip(classes, mu, strict=True):
        d = x[y == c] - m
        c_hat = d.T @ d / max(d.shape[0] - 1, 1)
        c_hat = (1 - pooled) * c_hat + pooled * cov_pooled
        c_hat = (1 - shrink) * c_hat + shrink * np.trace(c_hat) / c_hat.shape[0] * np.eye(
            c_hat.shape[0]
        )
        covs.append(c_hat)
    return {"classes": classes, "mu": mu, "covs": covs}


def qda_predict(model: dict[str, object], x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    classes = np.asarray(model["classes"])
    mu = model["mu"]
    covs = model["covs"]
    if not (isinstance(mu, list) and isinstance(covs, list)):
        raise ValueError("isinstance(mu, list) and isinstance(covs, list)")
    scores = np.stack(
        [_log_gauss(x, np.asarray(m), np.asarray(c)) for m, c in zip(mu, covs, strict=True)],
        axis=1,
    )
    return np.asarray(classes[np.argmax(scores, axis=1)])


def bench_discriminant(seed: int = 553) -> dict[str, float]:
    """SYNTHETIC: two classes with very different covariance
    orientations — QDA must beat LDA materially."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n = 160
    # class 0: elongated along +diag; class 1: elongated along −diag,
    # means close — linear boundary underfits the covariance twist
    th0, th1 = 0.6, -0.6
    rot0 = np.array([[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]])
    rot1 = np.array([[np.cos(th1), -np.sin(th1)], [np.sin(th1), np.cos(th1)]])
    x0 = rng.normal(0, [1.6, 0.25], (n, 2)) @ rot0.T + np.array([0.3, 0])
    x1 = rng.normal(0, [1.6, 0.25], (n, 2)) @ rot1.T + np.array([-0.3, 0])
    x = np.vstack([x0, x1])
    y = np.r_[np.zeros(n), np.ones(n)]
    perm = rng.permutation(2 * n)
    tr, te = perm[: 2 * n // 2], perm[2 * n // 2 :]
    m_lda = lda_fit(x[tr], y[tr])
    acc_lda = float((lda_predict(m_lda, x[te]) == y[te]).mean())
    out["synthetic_lda_acc"] = acc_lda
    m_qda = qda_fit(x[tr], y[tr], shrink=0.1, pooled=0.0)
    acc_qda = float((qda_predict(m_qda, x[te]) == y[te]).mean())
    out["synthetic_qda_acc"] = acc_qda
    if acc_qda < 0.8:
        raise ValueError(f"qda acc off: {acc_qda}")
    if acc_qda < acc_lda + 0.03:
        raise ValueError(f"qda not > lda: {acc_qda} vs {acc_lda}")
    return out
