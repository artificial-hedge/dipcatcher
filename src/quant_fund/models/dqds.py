"""dqds: differential quotient-difference-with-shifts for bidiag singular values (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 661


def dqds(B: np.ndarray, iters: int = 60, shift: float = 0.0) -> np.ndarray:
    """Iterate q/e recurrence; returns sorted singular values estimate."""
    n = B.shape[0]
    d = np.diag(B).copy()
    e = np.concatenate([np.diag(B, 1), [0.0]])
    q = d**2
    ee = e**2
    for _ in range(iters):
        q_new = np.zeros(n)
        e_new = np.zeros(n)
        d_var = q[0] - shift
        for i in range(n - 1):
            q_new[i] = d_var + ee[i]
            e_new[i] = ee[i] * q[i + 1] / q_new[i]
            d_var = d_var * q[i + 1] / q_new[i] - shift
        q_new[n - 1] = d_var
        q, ee = q_new, e_new
        if np.max(np.abs(ee)) < 1e-10:
            break
    return np.sort(np.sqrt(np.abs(q)))


def bench_dqds(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    rel_errs = []
    for _ in range(15):
        n = rng.randint(4, 8)
        B = np.diag(rng.rand(n) + 0.5) + np.diag(rng.rand(n - 1), 1)
        got = dqds(B)
        exp = np.sort(np.linalg.svd(B, compute_uv=False))
        rel_errs.append(float(np.linalg.norm(got - exp) / (np.linalg.norm(exp) + 1e-12)))
    return {
        "synthetic_dqds_rel_err": float(np.mean(rel_errs)),
        "synthetic_dqds_close": float(np.mean(rel_errs) < 0.1),
    }
