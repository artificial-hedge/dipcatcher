"""Multi-index / multilevel Monte Carlo (MIMC) — telescoping
estimators over a 2-D refinement lattice.

Generalizes MLMC: levels are multi-indices (l1, l2), the estimator is
Σ_l ΔP_l on a downward-closed index set, Δ defined by inclusion–
exclusion of coarse neighbors. Optimal index set chosen by the
profit heuristic (gain/cost ratio).
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def mimc_estimate(
    level_fn,
    levels: list[tuple[int, int]],
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Telescoping MIMC estimator.

    level_fn(l1, l2, rng) → a sample at refinement level (l1, l2).
    Returns (estimate, total_cost_units).
    """
    total = 0.0
    cost = 0.0
    for l1, l2 in levels:
        # ΔP = P(l1,l2) - P(l1-1,l2) - P(l1,l2-1) + P(l1-1,l2-1)
        terms = [(l1, l2, 1.0)]
        if l1 > 0:
            terms.append((l1 - 1, l2, -1.0))
        if l2 > 0:
            terms.append((l1, l2 - 1, -1.0))
        if l1 > 0 and l2 > 0:
            terms.append((l1 - 1, l2 - 1, 1.0))
        fine = np.array([level_fn(l1, l2, rng) for _ in range(n_samples)])
        delta = float(fine.mean())
        for a, b, sgn in terms[1:]:
            coarse = np.array([level_fn(a, b, rng) for _ in range(n_samples)])
            # couple same-seed estimate: reuse fine minus coarse via
            # independent sampling (simplified coupling)
            delta += sgn * float(coarse.mean())
        total += delta
        cost += (2.0 ** (l1 + l2)) * n_samples
    return total, cost


def profit_index_set(bias_fn, cost_fn, max_level: int, dim: int = 2) -> list[tuple[int, ...]]:
    """Greedy index set: add the level with best (bias reduction)/cost
    until exhausted. bias_fn(l) ≈ |E[P_l − P]| proxy."""
    set_ = [tuple([0] * dim)]
    frontier = list(set_)
    while len(set_) < 60:
        cands = []
        for lv in frontier:
            for k in range(dim):
                nll = list(lv)
                nll[k] += 1
                nl = tuple(nll)
                if max(nl) > max_level or nl in set_:
                    continue
                # downward-closed: all predecessors must be in set_
                ok = True
                for j in range(dim):
                    if nl[j] > 0:
                        pred = list(nl)
                        pred[j] -= 1
                        if tuple(pred) not in set_:
                            ok = False
                if not ok:
                    continue
                profit = bias_fn(nl) / cost_fn(nl)
                cands.append((profit, nl))
        if not cands:
            break
        cands.sort(key=lambda t: -t[0])
        frontier = [set_[-1]]
        set_.append(cands[0][1])
    return set_


def bench_mimc(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: 2-D refinement where convergence needs both indices —
    MIMC telescoping recovers the fine-grid expectation."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    # model: P(l1,l2) = E[f] + 2^-l1 * a + 2^-l2 * b + coupled 2^-(l1+l2)*c
    base = 1.5

    def level_fn(l1: int, l2: int, r: np.random.Generator) -> float:
        return base + 2.0**-l1 * 0.8 + 2.0**-l2 * 0.5 + 2.0 ** -(l1 + l2) * 0.3 + r.normal(0, 1e-6)

    # estimate P(4,4) via telescoping a downward-closed set
    levels = [(a, b) for a, b in itertools.product(range(5), repeat=2) if a + b <= 4]
    est, cost = mimc_estimate(level_fn, levels, 400, rng)
    fine = level_fn(4, 4, rng)
    out["synthetic_mimc_err"] = abs(est - fine)
    out["synthetic_mimc_cost"] = cost
    # single-index MLMC along l1 only misses the l2 bias — shows the
    # need for the multi-index correction
    levels_1d = [(a, 0) for a in range(5)]
    est1d, _ = mimc_estimate(level_fn, levels_1d, 400, rng)
    out["synthetic_mimc_1d_err"] = abs(est1d - fine)
    out["synthetic_mimc_better"] = float(out["synthetic_mimc_err"] < out["synthetic_mimc_1d_err"])
    return out


if __name__ == "__main__":
    print(bench_mimc())
