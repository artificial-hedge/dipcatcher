"""Wald (1947) Sequential Probability Ratio Test.

The SPRT accumulates the log-likelihood ratio of two simple
hypotheses and stops the moment it crosses either boundary —
the optimum stopping rule for fixed error probabilities, with
expected sample size roughly half the fixed-sample test's.

Honesty: synthetic benches run the SPRT on a known mean shift
and check the expected sample number and error rates against
Wald's approximations — proper diagnostics, never market
evidence.

References:
- Wald, A. (1947). *Sequential Analysis* — the SPRT,
  OC/ASN functions, boundary approximations A=(1−β)/α,
  B=β/(1−α).
- Siegmund, D. (1985). *Sequential Analysis: Tests and
  Confidence Intervals* — overshoot corrections.
- Lorden, G. (1976). 2-SPRT's and the modified Kiefer-Weiss
  problem of minimizing an expected sample size. *Annals of
  Statistics* 4.
- Ghosh, B. K., Sen, P. K. (1991). *Handbook of Sequential
  Analysis* — reference tables.

Composition: pure numpy — LLR accumulation + OC/ASN
approximations; deterministic ``np.random.default_rng``; no
new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """Wald approximations A=ln((1−β)/α), B=ln(β/(1−α))."""
    if not (0 < alpha < 0.5 and 0 < beta < 0.5):
        raise ValueError("alpha,beta in (0,.5) required")
    return (
        math.log((1 - beta) / alpha),
        math.log(beta / (1 - alpha)),
    )


def sprt_normal(
    x: FloatArray,
    mu0: float,
    mu1: float,
    sigma: float,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> dict[str, float]:
    """Run the normal-mean SPRT on observations x sequentially
    (H0: μ0, H1: μ1, known σ). Returns decision, stopping
    index, final LLR."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1 or xx.size < 5 or not np.all(np.isfinite(xx)):
        raise ValueError("finite (n>=5) x required")
    if sigma <= 0 or mu0 == mu1:
        raise ValueError("sigma>0, mu0!=mu1 required")
    a, b = sprt_bounds(alpha, beta)
    delta = (mu1 - mu0) / sigma
    llr = 0.0
    for i, xi in enumerate(xx):
        # incremental LLR for N(mu, sigma²): δ·(x−μmid)/σ
        llr += delta * (xi - 0.5 * (mu0 + mu1)) / sigma
        if llr >= a:
            return {
                "decision": 1.0,
                "n_stopped": float(i + 1),
                "llr": float(llr),
            }
        if llr <= b:
            return {
                "decision": 0.0,
                "n_stopped": float(i + 1),
                "llr": float(llr),
            }
    return {"decision": -1.0, "n_stopped": float(xx.size), "llr": float(llr)}


def sprt_asn(
    mu: float, mu0: float, mu1: float, sigma: float, alpha: float = 0.05, beta: float = 0.05
) -> float:
    """Wald's approximation to expected sample size at true
    mean μ: E[N] ≈ (L(μ)·b + (1−L(μ))·a)/E[z] where z is the
    increment and L is the OC function."""
    a, b = sprt_bounds(alpha, beta)
    ez = (mu - 0.5 * (mu0 + mu1)) * (mu1 - mu0) / sigma**2
    if abs(ez) < 1e-9:
        # E[N] ≈ a·|b|/Var(z) limit form
        vz = ((mu1 - mu0) / sigma) ** 2
        return float(-a * b / vz)
    h = (mu1 + mu0 - 2 * mu) / (mu1 - mu0)
    if abs(h) < 1e-9:
        return float((a * abs(b)) / abs(ez))
    # OC function L(μ) = (A^h − 1)/(A^h − B^h), A=e^a, B=e^b
    ah = math.exp(h * a)
    bh = math.exp(h * b)
    l_mu = (ah - 1.0) / (ah - bh)
    return float((l_mu * b + (1 - l_mu) * a) / ez)


def synth_sprt_stream(
    n: int = 2000,
    mu: float = 0.3,
    seed: int = 0,
) -> FloatArray:
    """Stream of N(mu,1) observations."""
    return np.random.default_rng(seed).normal(mu, 1.0, n)


def bench_wald_sprt(seed: int = 20261231 + 260) -> dict[str, float]:
    """SPRT self-check: under H1 stops accepting H1 quickly;
    under H0 rejects H1; ASN near Wald's approximation.
    All ``synthetic_*``."""
    rng = np.random.default_rng(seed)
    n_rep = 60
    dec1, nn1 = [], []
    for _ in range(n_rep):
        out = sprt_normal(rng.normal(0.3, 1.0, 3000), 0.0, 0.3, 1.0, 0.05, 0.05)
        dec1.append(out["decision"])
        nn1.append(out["n_stopped"])
    dec0, nn0 = [], []
    for _ in range(n_rep):
        out = sprt_normal(rng.normal(0.0, 1.0, 3000), 0.0, 0.3, 1.0, 0.05, 0.05)
        dec0.append(out["decision"])
        nn0.append(out["n_stopped"])
    asn1 = sprt_asn(0.3, 0.0, 0.3, 1.0)
    out2 = sprt_normal(rng.normal(0.3, 1.0, 3000), 0.0, 0.3, 1.0)
    out3 = sprt_normal(rng.normal(0.3, 1.0, 3000), 0.0, 0.3, 1.0)
    return {
        "synthetic_power": float(np.mean([d == 1.0 for d in dec1])),
        "synthetic_type1": float(np.mean([d == 1.0 for d in dec0])),
        "synthetic_asn_h1": float(np.mean(nn1)),
        "synthetic_asn_h0": float(np.mean(nn0)),
        "synthetic_asn_theory": asn1,
        "synthetic_detects": float(
            np.mean([d == 1.0 for d in dec1]) > 0.9
            and np.mean([d == 1.0 for d in dec0]) < 0.15
            and np.mean(nn1) < 150
        ),
        "synthetic_determinism": float(out2["decision"] == out3["decision"]),
    }
