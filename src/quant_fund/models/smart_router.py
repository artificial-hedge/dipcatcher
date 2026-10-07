"""ML-driven smart order router (Exec-Summary Feature 3). Fill-probability (SYNTHETIC)
and adverse-selection ("toxicity") models score candidate child orders on
each venue; a greedy marginal-cost allocator splits a parent order across
venues subject to displayed depth.

Synthetic bench: ML router vs round-robin on held-out tape — realized
cost, fill rate, toxicity incidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


@dataclass
class Venue:
    name: str
    depth: float  # displayed liquidity (fraction of parent fillable per slot)
    fee_bps: float
    latency_ms: float
    tox: float  # adverse-selection propensity of the venue's flow


_VENUES = [
    Venue("lit", 0.6, 0.3, 5.0, 0.10),
    Venue("midpt", 0.35, 0.15, 12.0, 0.35),
    Venue("darkA", 0.5, 0.05, 30.0, 0.55),
    Venue("darkB", 0.45, 0.05, 40.0, 0.70),
]


def _features(size_frac: float, v: Venue) -> FloatArray:
    return np.array(
        [1.0, size_frac, v.depth, v.fee_bps / 10.0, v.latency_ms / 50.0, size_frac / v.depth]
    )


def _fill_prob(x: float, rng: np.random.Generator, v: Venue) -> float:
    """x = venue's share of the parent order; fill decays toward depth cap."""
    sat = x / max(v.depth, 1e-9)
    base = 0.92 * np.exp(-v.latency_ms / 80.0) * (1.0 - 0.55 * min(sat, 1.0))
    return float(np.clip(base + 0.02 * rng.random(), 0.01, 0.99))


def _tox_move(x: float, rng: np.random.Generator, v: Venue) -> float:
    p = float(np.clip(v.tox * (0.4 + 1.6 * x), 0.02, 0.95))
    return float(rng.random() < p)


def synth_fills(
    venues: list[Venue], n: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    X, yf, yt, vi = [], [], [], []
    for _ in range(n):
        v = venues[int(rng.integers(len(venues)))]
        x = float(rng.uniform(0.02, 0.9))
        f = _features(x, v)
        X.append(f)
        vi.append(venues.index(v))
        yf.append(_fill_prob(x, rng, v))
        yt.append(_tox_move(x, rng, v))
    return np.array(X), np.array(yf), np.array(yt), np.array(vi)


def fit_ridge_logistic_target(X: FloatArray, y: FloatArray, lam: float = 1e-2) -> FloatArray:
    """Least-squares fit to a continuous target in [0,1] (pseudo-likelihood)."""
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.asarray(np.linalg.solve(A, X.T @ y))


def predict(w: FloatArray, x: FloatArray) -> FloatArray:
    return np.asarray(np.clip(x @ w, 0.0, 1.0))


def route_order(
    parent_size: float,
    venues: list[Venue],
    w_fill: FloatArray,
    w_tox: FloatArray,
    n_lots: int = 10,
    tox_penalty_bps: float = 24.0,
    unfilled_penalty_bps: float = 12.0,
) -> list[float]:
    """Greedy marginal-cost split of parent_size across venues."""
    alloc = np.zeros(len(venues))
    lot = parent_size / n_lots
    for _ in range(n_lots):
        best_v, best_c = -1, np.inf
        for i, v in enumerate(venues):
            if alloc[i] + lot > v.depth * parent_size:
                continue
            f = _features((alloc[i] + lot) / parent_size, v)
            pf = float(predict(w_fill, f))
            pt = float(predict(w_tox, f))
            c = v.fee_bps + (1 - pf) * unfilled_penalty_bps + pt * tox_penalty_bps
            if c < best_c:
                best_v, best_c = i, c
        if best_v < 0:
            break
        alloc[best_v] += lot
    return alloc.tolist()


def round_robin(parent_size: float, venues: list[Venue], n_lots: int = 10) -> list[float]:
    alloc = np.zeros(len(venues))
    lot = parent_size / n_lots
    for k in range(n_lots):
        i = k % len(venues)
        if alloc[i] + lot <= venues[i].depth * parent_size:
            alloc[i] += lot
        else:
            for j in range(len(venues)):
                if alloc[j] + lot <= venues[j].depth * parent_size:
                    alloc[j] += lot
                    break
    return alloc.tolist()


def eval_plan(
    alloc: list[float], venues: list[Venue], rng: np.random.Generator
) -> dict[str, float]:
    cost = 0.0
    filled = 0.0
    tox_hits = 0.0
    total = sum(alloc)
    for a, v in zip(alloc, venues, strict=True):
        if a <= 0:
            continue
        x = a / max(total, 1e-9)
        pf = _fill_prob(x, rng, v)
        fill = a * pf
        cost += a * v.fee_bps + (a - fill) * 12.0
        tox_hits += _tox_move(x, rng, v) * a * 24.0
        filled += fill
    return {
        "cost_bps": cost / max(total, 1e-9),
        "fill_rate": filled / max(total, 1e-9),
        "tox_cost_bps": tox_hits / max(total, 1e-9),
    }


def bench_smart_router(seed: int = 7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    X, yf, yt, _ = synth_fills(_VENUES, 4000, rng)
    w_fill = fit_ridge_logistic_target(X, yf)
    w_tox = fit_ridge_logistic_target(X, yt)
    rng_eval = np.random.default_rng(seed + 1)
    ml_cost, rr_cost, ml_fill, rr_fill = [], [], [], []
    ml_tox, rr_tox = [], []
    for _ in range(200):
        ml = route_order(1.0, _VENUES, w_fill, w_tox)
        rr = round_robin(1.0, _VENUES)
        r_ml = eval_plan(ml, _VENUES, rng_eval)
        r_rr = eval_plan(rr, _VENUES, rng_eval)
        ml_cost.append(r_ml["cost_bps"] + r_ml["tox_cost_bps"])
        rr_cost.append(r_rr["cost_bps"] + r_rr["tox_cost_bps"])
        ml_fill.append(r_ml["fill_rate"])
        rr_fill.append(r_rr["fill_rate"])
        ml_tox.append(r_ml["tox_cost_bps"])
        rr_tox.append(r_rr["tox_cost_bps"])
    # model skill on holdout
    Xh, yfh, yth, _ = synth_fills(_VENUES, 1000, rng_eval)
    fill_mae = float(np.mean(np.abs(predict(w_fill, Xh) - yfh)))
    return {
        "synthetic_sor_ml_cost_bps": float(np.mean(ml_cost)),
        "synthetic_sor_rr_cost_bps": float(np.mean(rr_cost)),
        "synthetic_sor_cost_margin_bps": float(np.mean(rr_cost) - np.mean(ml_cost)),
        "synthetic_sor_ml_fill_rate": float(np.mean(ml_fill)),
        "synthetic_sor_rr_fill_rate": float(np.mean(rr_fill)),
        "synthetic_sor_fill_mae": fill_mae,
        "synthetic_sor_ml_beats_rr_rate": float(np.mean(np.array(ml_cost) < np.array(rr_cost))),
        "synthetic_sor_ml_tox_bps": float(np.mean(ml_tox)),
        "synthetic_sor_rr_tox_bps": float(np.mean(rr_tox)),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_smart_router(), indent=1))
