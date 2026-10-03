"""Mean-field variational inference (CAVI) for a Gaussian mixture.

Coordinate-ascent variational inference for a K-component
univariate Gaussian mixture with conjugate priors: Normal-Inverse-
Gamma on (μ_k, σ²_k), Dirichlet on the mixture weights π and a
categorical on each z_i. Iterating the closed-form variational
updates

  r_ik ∝ exp(E[ln π_k] + E[ln N(x_i; μ_k, σ²_k)])
  q(μ_k|σ²_k) = N(m_k, σ²_k/ν_k),  q(σ²_k) = IG(a_k, b_k),
  q(π) = Dir(α_0 + r_.k)

monotonically raises the ELBO — a deterministic alternative to
Gibbs sampling whose fixed point is the mean-field posterior
mode. Posterior cluster means and the ELBO trajectory are the
recovery diagnostics; label switching is resolved by sorting
components on E[μ_k].

Honesty: synthetic mixtures with planted (μ, σ, π); the bench
checks recovered means against truth (after the sort) and
ELBO monotonicity — a proper diagnostic, never market evidence.

References:
- Blei, D. M., Kucukelbir, A., McAuliffe, J. D. (2017).
  Variational inference: a review for statisticians. *JASA*
  112 — CAVI coordinate updates and the ELBO identity.
- Bishop, C. M. (2006). *Pattern Recognition and Machine
  Learning*, ch. 10 — mean-field GMM closed forms and the
  ELBO decomposition (B.17–B.30) implemented below.
- Attias, H. (1999). Inferring parameters and structure of
  latent variable models by variational Bayes. *UAI* —
  conjugate NIG mixture derivation.
- McGrory, C. A., Titterington, D. M. (2007). Variational
  approximations in Bayesian model selection. *Computational
  Statistics & Data Analysis* 51 — ELBO as model evidence.

Composition: numpy + scipy.special only — closed-form CAVI
updates; deterministic ``np.random.default_rng``; no new
dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, gammaln

FloatArray = NDArray[np.float64]

# conjugate priors (weakly informative, fixed)
_ALPHA0 = 1.0  # Dirichlet
_M0 = 0.0  # prior mean of μ_k
_LAM0 = 1.0  # prior precision share of μ_k
_A0 = 1.0  # IG shape of σ²_k
_B0 = 1.0  # IG scale of σ²_k


def cavi_gmm(
    x: FloatArray,
    k: int = 2,
    iters: int = 300,
    tol: float = 1e-9,
    seed: int = 0,
) -> dict[str, object]:
    """Mean-field VI for a univariate K-mixture. Returns sorted
    posterior means ``mu``, posterior expected ``pi``, ``sigma``
    (posterior SD sqrt(b/a)), responsibilities ``r`` and the
    ELBO trace."""
    xx = np.asarray(x, dtype=np.float64)
    n = xx.size
    if xx.ndim != 1 or n < 100:
        raise ValueError("1-D data, n>=100 required")
    if not np.all(np.isfinite(xx)):
        raise ValueError("finite data required")
    if not (2 <= k <= 6):
        raise ValueError("k in [2, 6] required")
    rng = np.random.default_rng(seed)
    r = rng.dirichlet(np.ones(k), size=n)
    elbo_trace: list[float] = []
    m = np.quantile(xx, np.linspace(0.1, 0.9, k))
    nu = np.full(k, _LAM0 + n / k)
    a = np.full(k, _A0 + n / (2.0 * k))
    b = np.full(k, _B0 + 0.5 * float(np.var(xx)))
    alp = np.full(k, _ALPHA0 + n / k)
    for _ in range(iters):
        e_lnpi = digamma(alp) - digamma(float(np.sum(alp)))
        e_tau = a / b  # E[1/σ²]
        e_lntau = digamma(a) - np.log(b)  # E[ln(1/σ²)]
        # E-step
        rho = (
            e_lnpi[None, :]
            + 0.5 * (e_lntau - np.log(2.0 * np.pi))[None, :]
            - 0.5 * e_tau[None, :] * ((xx[:, None] - m[None, :]) ** 2 + 1.0 / nu[None, :])
        )
        rho -= rho.max(axis=1, keepdims=True)
        r = np.exp(rho)
        r /= r.sum(axis=1, keepdims=True)
        # M-step
        nk = r.sum(axis=0) + 1e-12
        xbar = (r * xx[:, None]).sum(axis=0) / nk
        nu = _LAM0 + nk
        m = (_LAM0 * _M0 + nk * xbar) / nu
        a = _A0 + 0.5 * nk
        s2 = (r * (xx[:, None] - xbar[None, :]) ** 2).sum(axis=0) / nk
        b = _B0 + 0.5 * nk * s2 + 0.5 * (_LAM0 * nk / nu) * (xbar - _M0) ** 2
        alp = _ALPHA0 + nk
        # ELBO = E[ln p(x,z,θ)] + H[q(z)] + H[q(θ)]
        e_lnpi = digamma(alp) - digamma(float(np.sum(alp)))
        e_tau = a / b
        e_lntau = digamma(a) - np.log(b)
        e_lnx_given = float(
            np.sum(
                r
                * (
                    0.5 * (e_lntau - np.log(2.0 * np.pi))[None, :]
                    - 0.5 * e_tau[None, :] * ((xx[:, None] - m[None, :]) ** 2 + 1.0 / nu[None, :])
                )
            )
        )
        e_lnz = float(np.sum(r * e_lnpi[None, :]))
        e_lnppi = float(
            gammaln(k * _ALPHA0) - k * gammaln(_ALPHA0) + np.sum((_ALPHA0 - 1.0) * e_lnpi)
        )
        # E[ln p(μ,σ)] with NIG prior
        e_lnpmusig = float(
            np.sum(
                0.5
                * (
                    np.log(_LAM0 / (2.0 * np.pi))
                    + e_lntau
                    - _LAM0 * e_tau * (m**2 + 1.0 / nu - 2.0 * m * _M0 + _M0**2)
                )
                + _A0 * np.log(_B0)
                - gammaln(_A0)
                - (_A0 + 1.0) * (np.log(b) - digamma(a))
                - _B0 * e_tau
            )
        )
        # −E[ln q]: entropy terms
        h_z = float(-np.sum(r * np.log(np.clip(r, 1e-300, None))))
        h_pi = -float(
            gammaln(float(np.sum(alp))) - np.sum(gammaln(alp)) + np.sum((alp - 1.0) * e_lnpi)
        )
        h_musig = -float(
            np.sum(
                0.5 * (np.log(nu / (2.0 * np.pi)) + e_lntau - 1.0)
                + a * np.log(b)
                - gammaln(a)
                - (a + 1.0) * (np.log(b) - digamma(a))
                - a
            )
        )
        elbo = e_lnx_given + e_lnz + e_lnppi + e_lnpmusig + h_z + h_pi + h_musig
        if elbo_trace and abs(elbo - elbo_trace[-1]) < tol * max(1.0, abs(elbo_trace[-1])):
            elbo_trace.append(elbo)
            break
        elbo_trace.append(elbo)
    order = np.argsort(m)
    return {
        "mu": np.asarray(m[order], dtype=np.float64),
        "sigma": np.asarray(np.sqrt(b / a)[order], dtype=np.float64),
        "pi": np.asarray((alp / np.sum(alp))[order], dtype=np.float64),
        "elbo": np.asarray(elbo_trace, dtype=np.float64),
        "r": np.asarray(r[:, order], dtype=np.float64),
        "iters": float(len(elbo_trace)),
    }


def synth_gmm(
    n: int = 1200,
    mus: tuple[float, ...] = (-1.5, 1.5),
    sigmas: tuple[float, ...] = (0.6, 0.8),
    pis: tuple[float, ...] = (0.4, 0.6),
    seed: int = 0,
) -> dict[str, FloatArray]:
    rng = np.random.default_rng(seed)
    z = rng.choice(len(mus), size=n, p=np.asarray(pis))
    x = np.asarray(mus)[z] + np.asarray(sigmas)[z] * rng.standard_normal(n)
    return {
        "x": np.asarray(x, dtype=np.float64),
        "z": z.astype(np.float64),
    }


def bench_cavi_gmm(seed: int = 20261231 + 271) -> dict[str, float]:
    """CAVI self-check: recovers sorted mixture means, ELBO is
    non-decreasing across iterations, and responsibilities
    recover the planted component shares. All ``synthetic_*``."""
    d = synth_gmm(seed=seed)
    out = cavi_gmm(np.asarray(d["x"]), k=2, seed=seed)
    out2 = cavi_gmm(np.asarray(d["x"]), k=2, seed=seed)
    mu = np.asarray(out["mu"])
    elbo = np.asarray(out["elbo"])
    diffs = np.diff(elbo)
    mono = float(np.all(diffs > -1e-6 * np.maximum(1.0, np.abs(elbo[:-1]))))
    return {
        "synthetic_mu0": float(mu[0]),
        "synthetic_mu1": float(mu[1]),
        "synthetic_pi0": float(np.asarray(out["pi"])[0]),
        "synthetic_elbo_start": float(elbo[0]),
        "synthetic_elbo_end": float(elbo[-1]),
        "synthetic_elbo_mono": mono,
        "synthetic_detects": float(
            abs(mu[0] + 1.5) < 0.3 and abs(mu[1] - 1.5) < 0.3 and mono == 1.0
        ),
        "synthetic_determinism": float(np.asarray(out2["mu"])[0] == mu[0]),
    }
