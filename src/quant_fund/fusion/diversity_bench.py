"""Diversity bench — how many independent bets does the stack actually make?

``attribution`` asks which members earn their weight; this asks whether
the members are *different*. The effective number of bets is the
eigenvalue participation of the member-residual correlation matrix:

    ENB = (sum λ_i)^2 / sum λ_i^2

ENB = 1 for identical members, = M for orthogonal members. A stack of M
heads with ENB ≈ 1.2 is one trade wearing five names — the fusion layer
should know that.

Bench: an identical-members stack (ENB → 1), a decorrelated stack
(ENB ≈ M), and a realistic stack (signal + noise mixes). Sealed
``diversity.v1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]


def effective_bets(residuals: Array) -> dict[str, Any]:
    """Eigenvalue-participation ENB of the member-residual correlation.

    ``residuals`` is (n_obs, n_members) of per-member errors — correlation
    of errors is what matters for risk-sharing, not correlation of
    predictions.
    """
    r = np.asarray(residuals, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] < r.shape[1] + 2:
        raise ValueError("residuals must be (n_obs, n_members) with n_obs > n_members+1")
    if not np.isfinite(r).all():
        raise ValueError("residuals must be finite")
    corr = np.corrcoef(r, rowvar=False)
    eig = np.linalg.eigvalsh(corr)
    eig = np.clip(eig, 0.0, None)
    s1, s2 = float(eig.sum()), float((eig**2).sum())
    enb = s1 * s1 / s2 if s2 > 0 else float(r.shape[1])
    return {
        "n_members": int(r.shape[1]),
        "eigenvalues": [float(v) for v in eig],
        "effective_bets": enb,
        "concentration": float(eig.max() / s1) if s1 > 0 else None,
    }


def diversity_bench(n: int = 400, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    m = 5

    def members(noise_sd: float, shared: float) -> Array:
        common = rng.normal(0, shared, n)
        # prediction errors = common noise + idiosyncratic noise
        return np.column_stack([common + rng.normal(0, noise_sd, n) for _ in range(m)])

    ident = members(1e-9, 1.0)  # all members share the same noise → ENB ~1
    ortho = np.column_stack([rng.normal(0, 1, n) for _ in range(m)])
    real = members(0.5, 0.7)  # realistic mix

    rep = {
        "identical_members": effective_bets(ident),
        "orthogonal_members": effective_bets(ortho),
        "realistic_mix": effective_bets(real),
    }
    verdict = (
        "ok"
        if rep["identical_members"]["effective_bets"] < 1.5
        and rep["orthogonal_members"]["effective_bets"] > m - 1.5
        else "weak"
    )
    payload: dict[str, Any] = {
        "kind": "diversity",
        "schema": "diversity.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "ENB ≈ 1 for cloned members, ≈ M for orthogonal members",
            "verdict": verdict,
            "enbs": {k: v["effective_bets"] for k, v in rep.items()},
        },
        "interpretation": rep,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
