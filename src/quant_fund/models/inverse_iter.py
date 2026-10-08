"""Inverse iteration with shift: targets interior eigenpairs by solving (SYNTHETIC)
(A - mu I)^-1 power steps — recovers an eigenvalue the plain power
method can't reach.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._eig_synth import sym_planted


def bench_inverse_iter(seed: int = 2997, iters: int = 60) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    A, lam = sym_planted(seed)
    lam_sorted = np.sort(lam)
    target = lam_sorted[5]  # interior eigenvalue
    mu = target + 0.15  # shift close to interior eigenvalue
    n = A.shape[0]
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    Minv = np.linalg.inv(A - mu * np.eye(n))
    for _ in range(iters):
        w = Minv @ v
        v = w / np.linalg.norm(w)
    lam_hat = float(v @ A @ v)
    # Jacobi-Davidson-ish check: distance to nearest true eigenvalue
    err = float(np.min(np.abs(lam_sorted - lam_hat)))
    return {
        "synthetic_inviter_eig_err": err,
        "synthetic_inviter_eig_hat": lam_hat,
        "synthetic_torch_available": 0.0,
    }
