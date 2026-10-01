"""selective_inference — Lee–Sun polyhedral CIs after argmin selection.

winner_curse (PR #385) corrects the optimism of the argmin on the loss
table via bootstrap; this lane implements the *exact* conditional
approach: after observing which head won, inference on the winner's
true mean must condition on the selection event.

Lee et al. (2016): if L ~ N(μ, Σ) and the selection is {A·L ≤ b}
(the argmin constraints are linear — winner ≤ every other head), then
ηᵀL | {A·L ≤ b} is a *truncated normal* for any contrast η. The pivot
F^η_{μ,σ²}([V⁻,V⁺])(ηᵀL) ~ Unif(0,1) under the conditional law; the CI
for ηᵀμ inverts it. Valid coverage conditional on selection — no
winner's curse inflation.

Bench: synthetic MVN loss tables with a planted ordering; check the
conditional CI covers the winner's true mean at ≈ the nominal rate,
while the naive (unconditioned) z-interval undercovers as the theory
predicts. All SYNTHETIC; sealed receipt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

SELECTIVE_SCHEMA = "selective_inference.v1"


def argmin_polytope(k_winner: int, k: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Linear constraints {A·z ≤ b} encoding ``argmin(z) == k_winner``.

    For each j ≠ winner: z_j - z_w ≥ 0  →  -z_j + z_w ≤ 0.
    """
    A = np.zeros((k - 1, k))
    b = np.zeros(k - 1)
    rows = [j for j in range(k) if j != k_winner]
    for r, j in enumerate(rows):
        A[r, k_winner] = 1.0
        A[r, j] = -1.0
    return A, b


def _tn_cdf(x: float, mu: float, sigma: float, lo: float, hi: float) -> float:
    """CDF of N(mu, sigma²) truncated to [lo, hi], evaluated at x.

    Log-CDF form: plain Φ differences underflow when both standardized
    edges sit deep in one tail (e.g. mu ≫ hi on a one-sided truncation).
    """
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError(f"sigma must be positive finite, got {sigma}")
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    z = (x - mu) / sigma
    lc_a, lc_b, lc_z = norm.logcdf(a), norm.logcdf(b), norm.logcdf(min(z, b))
    # (Φ(z)−Φ(a))/(Φ(b)−Φ(a)) in factored log form
    num = math.exp(lc_z) - math.exp(lc_a)
    den = math.exp(lc_b) - math.exp(lc_a)
    if den <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, num / den)))


def truncation_interval(
    z: NDArray[np.float64],
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    eta: NDArray[np.float64],
    Sigma: NDArray[np.float64],
) -> tuple[float, float]:
    """[V⁻, V⁺]: the range of ηᵀz consistent with the selection.

    Lee–Sun–Taylor: for each constraint row i, write z = z_⊥ + t·c with
    c = Ση/ηᵀΣη; then A_i·z ≤ b_i is linear in t.
    """
    eta_Sigma = Sigma @ eta
    c = eta_Sigma / float(eta @ eta_Sigma)
    z_perp = z - c * float(eta @ z)
    lo, hi = -np.inf, np.inf
    for i in range(A.shape[0]):
        a_i = float(A[i] @ c)
        rhs_i = b[i] - float(A[i] @ z_perp)
        if abs(a_i) < 1e-12:
            if rhs_i < 0.0:
                return (np.inf, -np.inf)  # infeasible
            continue
        bound = rhs_i / a_i
        if a_i > 0:
            hi = min(hi, bound)
        else:
            lo = max(lo, bound)
    return float(lo), float(hi)


@dataclass(frozen=True)
class SelectiveCI:
    """Conditional CI for the winner's mean after argmin selection."""

    winner: int
    lo: float
    hi: float
    naive_lo: float
    naive_hi: float
    v_minus: float
    v_plus: float
    feasible: bool


def selective_ci(
    means: NDArray[np.float64],
    cov: NDArray[np.float64],
    *,
    alpha: float = 0.10,
) -> SelectiveCI:
    """(1−α) conditional CI for μ_{argmin} via the truncated-Gaussian pivot.

    Uses the equal-tailed pivot: solve F(ηᵀz; μ, V⁻,V⁺) = 1−α/2 and α/2
    for μ by bisection on the pivot's monotone-in-μ property.
    """
    means = np.asarray(means, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    k = means.size
    w = int(np.argmin(means))
    A, b = argmin_polytope(w, k)
    eta = np.zeros(k)
    eta[w] = 1.0
    sigma2 = float(eta @ cov @ eta)
    sigma = math.sqrt(sigma2)
    z_obs = float(means[w])
    v_m, v_p = truncation_interval(means, A, b, eta, cov)
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    naive = (z_obs - z_crit * sigma, z_obs + z_crit * sigma)
    if v_m >= v_p:
        return SelectiveCI(
            winner=w,
            lo=float("nan"),
            hi=float("nan"),
            naive_lo=naive[0],
            naive_hi=naive[1],
            v_minus=v_m,
            v_plus=v_p,
            feasible=False,
        )

    def pivot(mu: float) -> float:
        return _tn_cdf(z_obs, mu, sigma, v_m, v_p)

    # pivot(mu) = F_{TN(mu,sigma,[V-,V+])}(z_obs) is decreasing in mu.
    # Bracket adaptively: expand left while pivot > target (bound below),
    # right while pivot < target. One-sided truncation (V- = -inf) can
    # leave one endpoint at -inf — report it honestly rather than clamp.
    def bracket_low(target: float) -> tuple[float, bool]:
        lo_b = z_obs - 8.0 * sigma
        for _ in range(60):
            if pivot(lo_b) >= target:
                return lo_b, True
            lo_b -= 8.0 * sigma
            if lo_b < -1e6 * sigma:
                return float("-inf"), False
        return float("-inf"), False

    def bracket_high(target: float) -> tuple[float, bool]:
        hi_b = z_obs + 8.0 * sigma
        for _ in range(60):
            if pivot(hi_b) <= target:
                return hi_b, True
            hi_b += 8.0 * sigma
            if hi_b > 1e6 * sigma:
                return float("inf"), False
        return float("inf"), False

    # Degenerate boundary: z_obs sits exactly at V+ (a tie with the
    # runner-up) — pivot ≡ 1, the conditional CI is uninformative.
    if v_p - z_obs < 1e-9 * max(1.0, abs(z_obs)):
        return SelectiveCI(
            winner=w,
            lo=float("-inf"),
            hi=float("inf"),
            naive_lo=naive[0],
            naive_hi=naive[1],
            v_minus=v_m,
            v_plus=v_p,
            feasible=True,
        )

    def solve(target: float) -> float:
        lo_b, ok_l = bracket_low(target)
        if not ok_l:
            return float("-inf") if target > 0.5 else float("inf")
        hi_b, ok_r = bracket_high(target)
        if not ok_r:
            return float("inf") if target > 0.5 else float("-inf")
        for _ in range(90):
            m = 0.5 * (lo_b + hi_b)
            if pivot(m) > target:
                lo_b = m
            else:
                hi_b = m
        return 0.5 * (lo_b + hi_b)

    return SelectiveCI(
        winner=w,
        lo=solve(1.0 - alpha / 2.0),
        hi=solve(alpha / 2.0),
        naive_lo=naive[0],
        naive_hi=naive[1],
        v_minus=v_m,
        v_plus=v_p,
        feasible=True,
    )


def _simulate(
    rng: np.random.Generator,
    *,
    k: int,
    n: int,
    true_means: NDArray[np.float64],
    cov_scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Draw k per-head mean losses from N(true_means, cov_scale·I/n) with
    cross-head correlation from shared noise (one shared factor)."""
    rho = 0.4
    Sigma = np.full((k, k), rho * cov_scale / n)
    np.fill_diagonal(Sigma, cov_scale / n)
    return rng.multivariate_normal(true_means, Sigma), Sigma


def selective_bench(
    *,
    n_reps: int = 300,
    k: int = 6,
    n_origins: int = 120,
    alpha: float = 0.10,
    seed: int = 0,
) -> dict[str, Any]:
    """Coverage: conditional vs naive CI for the winner's true mean."""
    rng = np.random.default_rng(seed)
    true_means = np.array([0.05, 0.06, 0.065, 0.07, 0.075, 0.09])[:k]
    if true_means.size < k:
        true_means = np.linspace(0.05, 0.05 + 0.01 * k, k)
    cond_cov: list[bool] = []
    naive_cov: list[bool] = []
    widths_cond: list[float] = []
    widths_naive: list[float] = []
    n_infeasible = 0
    for _ in range(n_reps):
        means, Sigma = _simulate(rng, k=k, n=n_origins, true_means=true_means, cov_scale=1.0)
        ci = selective_ci(means, Sigma, alpha=alpha)
        if not ci.feasible:
            n_infeasible += 1
            continue
        truth = float(true_means[ci.winner])
        cond_cov.append(ci.lo <= truth <= ci.hi)
        naive_cov.append(ci.naive_lo <= truth <= ci.naive_hi)
        widths_cond.append(ci.hi - ci.lo)
        widths_naive.append(ci.naive_hi - ci.naive_lo)
    cov_cond = float(np.mean(cond_cov)) if cond_cov else float("nan")
    cov_naive = float(np.mean(naive_cov)) if naive_cov else float("nan")
    payload: dict[str, Any] = {
        "schema": SELECTIVE_SCHEMA,
        "kind": "selective_inference",
        "n_reps": n_reps,
        "k": k,
        "n_origins": n_origins,
        "alpha": alpha,
        "coverage_conditional": cov_cond,
        "coverage_naive": cov_naive,
        "mean_width_conditional": float(np.mean(widths_cond)) if widths_cond else float("nan"),
        "mean_width_naive": float(np.mean(widths_naive)) if widths_naive else float("nan"),
        "n_infeasible": n_infeasible,
        "conditional_calibrated": bool(
            np.isfinite(cov_cond) and abs(cov_cond - (1.0 - alpha)) < 0.06
        ),
        "interpretation": (
            "Lee-Sun polyhedral CI for the argmin winner's mean: must "
            "hold ~1-alpha coverage conditional on selection; the naive "
            "interval's undercovering is the winner's-curse signature"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["payload_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
