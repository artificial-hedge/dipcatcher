"""Derivative-free metaheuristic optimizers: simulated annealing, (SYNTHETIC)
differential evolution, particle swarm, a simple genetic
algorithm, and NSGA-II nondominated sorting for multiobjective
search. Synthetic benches gate multimodal convergence."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def simulated_annealing(
    f: Callable[[FloatArray], float],
    x0: FloatArray,
    t0: float = 1.0,
    alpha: float = 0.95,
    step: float = 0.5,
    it: int = 3000,
    seed: int = 0,
) -> dict[str, object]:
    """Simulated annealing (Kirkpatrick et al. 1983): Gaussian
    proposals with Metropolis acceptance on a geometric
    cooling schedule."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=np.float64).copy()
    fx = float(f(x))
    best_x, best_f = x.copy(), fx
    t = t0
    for _ in range(it):
        cand = x + rng.normal(scale=step, size=x.shape)
        fc = float(f(cand))
        if fc < fx or rng.uniform() < np.exp(-(fc - fx) / max(t, 1e-12)):
            x, fx = cand, fc
            if fx < best_f:
                best_x, best_f = x.copy(), fx
        t *= alpha
    return {"x": best_x, "f": best_f}


def differential_evolution(
    f: Callable[[FloatArray], float],
    lo: FloatArray,
    hi: FloatArray,
    npop: int = 40,
    factor: float = 0.8,
    cr: float = 0.9,
    it: int = 80,
    seed: int = 0,
) -> dict[str, object]:
    """DE/rand/1/bin (Storn & Price 1997): donor mutation plus
    binomial crossover with greedy selection."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    d = lo.size
    pop = lo + (hi - lo) * rng.uniform(size=(npop, d))
    fit = np.array([f(p) for p in pop])
    for _ in range(it):
        for i in range(npop):
            idx = rng.choice([j for j in range(npop) if j != i], 3, replace=False)
            a, b, c = pop[idx]
            mutant = np.clip(a + factor * (b - c), lo, hi)
            mask = rng.uniform(size=d) < cr
            mask[rng.integers(d)] = True
            trial = np.where(mask, mutant, pop[i])
            ft = float(f(trial))
            if ft < fit[i]:
                pop[i], fit[i] = trial, ft
    k = int(np.argmin(fit))
    return {"x": pop[k].copy(), "f": float(fit[k]), "pop": pop}


def particle_swarm(
    f: Callable[[FloatArray], float],
    lo: FloatArray,
    hi: FloatArray,
    npop: int = 40,
    c1: float = 1.5,
    c2: float = 1.5,
    w: float = 0.7,
    it: int = 80,
    seed: int = 0,
) -> dict[str, object]:
    """PSO (Kennedy & Eberhart 1995): inertia + cognitive +
    social velocity updates with position clamping."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    d = lo.size
    pos = lo + (hi - lo) * rng.uniform(size=(npop, d))
    vel = rng.uniform(-1, 1, (npop, d)) * 0.1 * (hi - lo)
    pbest = pos.copy()
    pfit = np.array([f(p) for p in pos])
    g = pbest[int(np.argmin(pfit))].copy()
    for _ in range(it):
        vel = (
            w * vel
            + c1 * rng.uniform(size=(npop, d)) * (pbest - pos)
            + c2 * rng.uniform(size=(npop, d)) * (g - pos)
        )
        pos = np.clip(pos + vel, lo, hi)
        fit = np.array([f(p) for p in pos])
        better = fit < pfit
        pbest[better] = pos[better]
        pfit[better] = fit[better]
        g = pbest[int(np.argmin(pfit))].copy()
    return {"x": g.copy(), "f": float(pfit.min()), "pop": pos}


def genetic_algorithm(
    f: Callable[[FloatArray], float],
    lo: FloatArray,
    hi: FloatArray,
    npop: int = 60,
    elite: int = 4,
    sigma: float = 0.1,
    it: int = 80,
    seed: int = 0,
) -> dict[str, object]:
    """Real-coded GA: tournament selection, blend (BLX-0.5)
    crossover, Gaussian mutation, elitist replacement."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    d = lo.size
    pop = lo + (hi - lo) * rng.uniform(size=(npop, d))
    fit = np.array([f(p) for p in pop])
    for _ in range(it):
        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        newpop = [pop[i].copy() for i in range(elite)]
        while len(newpop) < npop:
            i1 = rng.integers(npop)
            i2 = rng.integers(npop)
            p1 = pop[min(i1, i2)] if fit[min(i1, i2)] < fit[max(i1, i2)] else pop[max(i1, i2)]
            i3 = rng.integers(npop)
            i4 = rng.integers(npop)
            p2 = pop[min(i3, i4)] if fit[min(i3, i4)] < fit[max(i3, i4)] else pop[max(i3, i4)]
            u = rng.uniform(-0.5, 1.5, d)
            child = p1 + u * (p2 - p1) + sigma * rng.normal(size=d) * (hi - lo) * 0.1
            newpop.append(np.clip(child, lo, hi))
        pop = np.asarray(newpop[:npop])
        fit = np.array([f(p) for p in pop])
    k = int(np.argmin(fit))
    return {"x": pop[k].copy(), "f": float(fit[k])}


def _dominates(a: FloatArray, b: FloatArray) -> bool:
    """Pareto dominance for minimization."""
    return bool(np.all(a <= b) and np.any(a < b))


def nsga2_sort(obj: FloatArray) -> dict[str, object]:
    """NSGA-II (Deb et al. 2002): fast nondominated sorting
    with crowding-distance assignment."""
    obj = np.asarray(obj, dtype=np.float64)
    n, m = obj.shape
    S: list[list[int]] = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(obj[p], obj[q]):
                S[p].append(q)
            elif _dominates(obj[q], obj[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    rank = np.full(n, -1)
    crowd = np.zeros(n)
    for fi, fr in enumerate(fronts):
        for p in fr:
            rank[p] = fi
        for k in range(m):
            order = np.argsort(obj[fr, k])
            fr_sorted = [fr[j] for j in order]
            crowd[fr_sorted[0]] = crowd[fr_sorted[-1]] = np.inf
            span = obj[fr_sorted[-1], k] - obj[fr_sorted[0], k]
            span = span if span > 0 else 1.0
            for j in range(1, len(fr) - 1):
                crowd[fr_sorted[j]] += (obj[fr_sorted[j + 1], k] - obj[fr_sorted[j - 1], k]) / span
    return {"fronts": fronts, "rank": rank, "crowding": crowd}


def bench_metaheuristics(seed: int = 540) -> dict[str, float]:
    """SYNTHETIC: multimodal Rastrigin recovery by SA/DE/PSO/GA
    and NSGA-II front ranking on a planted two-objective set."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    lo = np.full(4, -5.12)
    hi = np.full(4, 5.12)

    def rastrigin(x: FloatArray) -> float:
        x = np.asarray(x)
        return float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

    x0 = rng.uniform(lo, hi)
    sa = simulated_annealing(rastrigin, x0, t0=2.0, alpha=0.995, it=4000, seed=seed)
    out["synthetic_sa_rastrigin_f"] = float(np.asarray(sa["f"]))
    de = differential_evolution(rastrigin, lo, hi, npop=50, it=150, seed=seed)
    out["synthetic_de_rastrigin_f"] = float(np.asarray(de["f"]))
    ps = particle_swarm(rastrigin, lo, hi, npop=50, it=150, seed=seed)
    out["synthetic_pso_rastrigin_f"] = float(np.asarray(ps["f"]))
    ga = genetic_algorithm(rastrigin, lo, hi, npop=80, it=150, seed=seed)
    out["synthetic_ga_rastrigin_f"] = float(np.asarray(ga["f"]))
    if out["synthetic_de_rastrigin_f"] > 5.0:
        raise ValueError(f"DE off Rastrigin: {de['f']}")
    if out["synthetic_pso_rastrigin_f"] > 5.0:
        raise ValueError(f"PSO off Rastrigin: {ps['f']}")
    if out["synthetic_sa_rastrigin_f"] > 20.0:
        raise ValueError(f"SA off Rastrigin: {sa['f']}")
    if out["synthetic_ga_rastrigin_f"] > 30.0:
        raise ValueError(f"GA off Rastrigin: {ga['f']}")
    # ZDT1-style: front must separate planted optimal f2 = 1-sqrt(f1)
    f1 = np.linspace(0, 1, 50)
    front_true = np.c_[f1, 1 - np.sqrt(f1)]
    dominated = np.c_[rng.uniform(0.5, 1.5, 30), 1 - np.sqrt(rng.uniform(0.5, 1.0, 30)) + 0.5]
    pts = np.vstack([front_true, dominated])
    r = nsga2_sort(pts)
    fronts = r["fronts"]
    if not (isinstance(fronts, list)):
        raise ValueError("isinstance(fronts, list)")
    front0 = [int(i) for i in fronts[0]]
    in_first = set(front0)
    out["synthetic_nsga_front0_size"] = float(len(front0))
    n_true_in = len([i for i in front0 if i < 50])
    out["synthetic_nsga_front0_true"] = float(n_true_in)
    if n_true_in < 48 or any(i >= 50 for i in in_first):
        raise ValueError(f"nsga front off: {n_true_in} of front-0")
    return out
