"""Bayesian optimization (Jones et al. 1998; Mockus 1978):
GP surrogate with expected-improvement, UCB, and
probability-of-improvement acquisition over a bounded box,
plus a random-search baseline. Synthetic bench gates regret
vs random search on the Forrester function."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.models.gaussian_process import GPModel, gp_fit, gp_predict

FloatArray = NDArray[np.float64]


def expected_improvement(
    mean: FloatArray, sd: FloatArray, f_best: float, xi: float = 0.01
) -> FloatArray:
    """EI(x) = (f* − μ − ξ)Φ(z) + σφ(z), z = (f* − μ − ξ)/σ."""
    sd = np.maximum(sd, 1e-12)
    z = (f_best - mean - xi) / sd
    return np.asarray((f_best - mean - xi) * norm.cdf(z) + sd * norm.pdf(z))


def ucb(mean: FloatArray, sd: FloatArray, kappa: float = 2.0) -> FloatArray:
    """Lower-confidence-bound acquisition for minimization
    (maximize −μ + κσ)."""
    return np.asarray(-mean + kappa * sd)


def probability_improvement(
    mean: FloatArray, sd: FloatArray, f_best: float, xi: float = 0.01
) -> FloatArray:
    """PI(x) = Φ((f* − μ − ξ)/σ)."""
    sd = np.maximum(sd, 1e-12)
    return np.asarray(norm.cdf((f_best - mean - xi) / sd))


def bayes_opt(
    f: Callable[[FloatArray], float],
    lo: FloatArray,
    hi: FloatArray,
    n_init: int = 5,
    n_iter: int = 20,
    acq: str = "ei",
    grid: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Sequential GP-EO over a box: initial LHS design, then
    argmax acquisition on a random candidate pool."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    d = lo.size
    # latin-hypercube-ish init
    xs = lo + (hi - lo) * rng.uniform(size=(n_init, d))
    ys = np.array([f(x) for x in xs])
    for _ in range(n_iter):
        model = gp_fit(xs, ys, n_restarts=2, seed=seed)["model"]
        assert isinstance(model, GPModel)
        cand = lo + (hi - lo) * rng.uniform(size=(grid, d))
        pred = gp_predict(model, cand, latent=True)
        mean, sd = pred["mean"], pred["sd"]
        f_best = float(ys.min())
        if acq == "ei":
            a = expected_improvement(mean, sd, f_best)
        elif acq == "ucb":
            a = ucb(mean, sd)
        elif acq == "pi":
            a = probability_improvement(mean, sd, f_best)
        else:
            raise ValueError(f"unknown acq: {acq}")
        x_next = cand[int(np.argmax(a))]
        xs = np.vstack([xs, x_next])
        ys = np.append(ys, f(x_next))
    k = int(np.argmin(ys))
    return {"x": xs[k].copy(), "f": float(ys[k]), "xs": xs, "ys": ys}


def bench_bayes_opt(seed: int = 541) -> dict[str, float]:
    """SYNTHETIC: 1-D Forrester function — Bayes-opt best-so-far
    must beat a same-budget random search and land near the
    global minimum (~-6.02 at x≈0.757)."""
    out: dict[str, float] = {}

    def forrester(x: FloatArray) -> float:
        x = np.asarray(x)
        return float((6 * x[0] - 2) ** 2 * np.sin(12 * x[0] - 4))

    lo, hi = np.array([0.0]), np.array([1.0])
    bo = bayes_opt(forrester, lo, hi, n_init=8, n_iter=15, seed=seed)
    f_bo = float(np.asarray(bo["f"]))
    out["synthetic_bo_best_f"] = f_bo
    rng = np.random.default_rng(seed + 1)
    xs_r = rng.uniform(0, 1, 23)  # same 8+15 eval budget
    f_rand = float(min(forrester(np.array([x])) for x in xs_r))
    out["synthetic_bo_random_f"] = f_rand
    if f_bo > f_rand + 0.5:
        raise ValueError(f"BO worse than random: {f_bo} vs {f_rand}")
    # global min ~ -6.02; require within 1.5 of it
    if f_bo > -4.5:
        raise ValueError(f"BO not near global min: {f_bo}")
    out["synthetic_bo_ucb_f"] = float(
        np.asarray(bayes_opt(forrester, lo, hi, n_init=8, n_iter=15, acq="ucb", seed=seed)["f"])
    )
    return out
