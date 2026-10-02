"""Economic calendar & news impact engine (Exec-Summary item).
Releases carry (name, consensus, actual); surprise z-scores feed a
ridge regression estimating per-event price impact and half-life.

Synthetic bench: coefficient recovery error and out-of-sample R^2 on
surprise -> move mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


@dataclass
class Release:
    name: str
    consensus: float
    actual: float
    hist_std: float
    impact_true: float  # latent coefficient (bps of move per sigma surprise)


_EVENTS = ["cpi", "nfp", "fomc", "retail", "pmi"]


def synth_calendar(
    n_events: int, rng: np.random.Generator
) -> tuple[list[Release], FloatArray, FloatArray]:
    """Returns releases, surprise z, realized move (bps)."""
    base_impact = {"cpi": 18.0, "nfp": 25.0, "fomc": 30.0, "retail": 8.0, "pmi": 10.0}
    rels: list[Release] = []
    zs = np.zeros(n_events)
    mv = np.zeros(n_events)
    for i in range(n_events):
        name = _EVENTS[i % len(_EVENTS)]
        cons = rng.uniform(0, 3)
        hist_std = 0.4
        actual = cons + hist_std * rng.standard_normal()
        z = (actual - cons) / hist_std
        imp = base_impact[name]
        zs[i] = z
        mv[i] = imp * z + 6.0 * rng.standard_normal()
        rels.append(Release(name, cons, actual, hist_std, imp))
    return rels, zs, mv


def fit_impact(
    names: list[str], zs: FloatArray, mv: FloatArray, lam: float = 1e-2
) -> dict[str, float]:
    """Per-event ridge coefficient mapping surprise z -> move bps."""
    coefs: dict[str, float] = {}
    for ev in _EVENTS:
        idx = [i for i, n in enumerate(names) if n == ev]
        z, m = zs[idx], mv[idx]
        coefs[ev] = float(np.sum(z * m) / (np.sum(z * z) + lam))
    return coefs


def predict_impact(name: str, z: float, coefs: dict[str, float]) -> float:
    return coefs.get(name, 0.0) * z


def half_life(moves: FloatArray, horizon: int = 10) -> float:
    """Simple AR(1) half-life of a move series."""
    x = moves[:-1]
    y = moves[1:]
    if np.std(x) < 1e-9:
        return 0.0
    b = float(np.sum(x * y) / np.sum(x * x))
    b = min(abs(b), 0.99)
    return float(np.log(0.5) / np.log(b))


def bench_econ_calendar(seed: int = 7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    rels, zs, mv = synth_calendar(500, rng)
    names = [r.name for r in rels]
    tr = slice(0, 350)
    te = slice(350, 500)
    coefs = fit_impact(names[tr], zs[tr], mv[tr])
    pred = np.array([predict_impact(names[i], zs[i], coefs) for i in range(*te.indices(len(zs)))])
    err = pred - mv[te]
    r2 = 1.0 - float(np.sum(err**2)) / float(np.sum((mv[te] - mv[te].mean()) ** 2))
    coef_err = np.mean(
        [abs(coefs[e] - rels[i].impact_true) / rels[i].impact_true for i, e in enumerate(_EVENTS)]
    )
    sign_hit = float(np.mean(np.sign(pred) == np.sign(mv[te])))
    return {
        "synthetic_econ_impact_oos_r2": r2,
        "synthetic_econ_coef_rel_err": float(coef_err),
        "synthetic_econ_sign_accuracy": sign_hit,
        "synthetic_econ_half_life_steps": half_life(mv),
        "synthetic_econ_events": float(len(rels)),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_econ_calendar(), indent=1))
