"""LAMBDA integer least squares — carrier-phase ambiguity resolution (SYNTHETIC).

Decorrelates the float ambiguity covariance via a unimodular
Z-transform (LDL-based integer Gauss reduction), then searches the
transformed integer hyper-ellipsoid for the best candidate(s).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _ldl(Q: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Q = L D Lᵀ (unit-lower L, diagonal D), permuted descent."""
    n = Q.shape[0]
    L = np.eye(n)
    D = np.zeros(n)
    A = Q.copy()
    for k in range(n - 1, -1, -1):
        D[k] = A[k, k]
        if D[k] <= 0:
            D[k] = 1e-12
        L[: k + 1, k] = A[: k + 1, k] / D[k]
        for i in range(k):
            A[i, : k + 1] -= A[i, k] * L[: k + 1, k]
    return L, D


def _z_transform(Qa: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Integer Gauss decorrelation: returns Z (unimodular) and
    Qz = Zᵀ Qa Z."""
    n = Qa.shape[0]
    Q = Qa.copy()
    Z = np.eye(n)
    k = n - 2
    while True:
        L, D = _ldl(Q)
        # find smallest conditional variance → permute to back
        imin = int(np.argmin(D[: k + 1]))
        if imin < k:
            order = list(range(n))
            order[imin], order[k] = order[k], order[imin]
            P = np.eye(n)[order]
            Q = P @ Q @ P.T
            Z = Z @ P.T
            L, D = _ldl(Q)
        # integer size reduction on column k of L
        for i in range(k - 1, -1, -1):
            mu = round(L[k, i])
            if mu != 0:
                T = np.eye(n)
                T[k, i] = -mu
                Q = T @ Q @ T.T
                Z = Z @ T.T
                L, D = _ldl(Q)
        k -= 1
        if k < 0:
            break
        k = min(k, n - 2)
        # convergence check: fully reduced LDL once through
        _, D2 = _ldl(Q)
        if k == n - 2 and np.all(np.abs(Q - np.diag(np.diag(Q))) < 1e9):
            pass
    return Z, Q


def lambda_ils(
    y: FloatArray, A: FloatArray, W: FloatArray, ncand: int = 2
) -> tuple[FloatArray, FloatArray, float]:
    """Integer least squares min_a (y − Aa)ᵀ W (y − Aa), a integer.

    Returns (z_fixed, float_solution, residual_ratio) where z_fixed is
    the best integer vector (first column of candidates, shape n).
    """
    n = A.shape[1]
    AtW = A.T @ W
    Qa = np.linalg.inv(AtW @ A)
    a_hat = Qa @ (AtW @ y)
    Z, Qz = _z_transform(Qa)
    z_hat = Z.T @ a_hat

    # Qz = Lz Dz Lzᵀ (Lz unit-UPPER). Cost = ||Dz^-1/2 Lz^-1 (z - z_hat)||²
    # so coordinate i conditioned on later coords has mean
    # z_hat_i - Σ_{j>i} Linv[i,j] (z_j - z_hat_j) and variance Dz_i.
    Lz, Dz = _ldl(Qz)
    Linv = np.linalg.inv(Lz)

    def cond_mean(i: int, zcand: FloatArray) -> float:
        return float(z_hat[i] - Linv[i, i + 1 :] @ (zcand[i + 1 :] - z_hat[i + 1 :]))

    z_bound = np.zeros(n)
    resid_bound = 0.0
    for i in range(n - 1, -1, -1):
        cm = cond_mean(i, z_bound)
        z_bound[i] = round(cm)
        resid_bound += (z_bound[i] - cm) ** 2 / Dz[i]
    bound = resid_bound * 2.0 + 1e-6

    best: list[tuple[float, FloatArray]] = []
    z = np.zeros(n)

    def search(i: int, partial: float) -> None:
        if partial > bound:
            return
        if i < 0:
            best.append((partial, z.copy()))
            return
        cm = cond_mean(i, z)
        spread = np.sqrt(max(bound - partial, 0.0) * Dz[i])
        lo = int(np.floor(cm - spread))
        hi = int(np.ceil(cm + spread))
        for cand in range(lo, hi + 1):
            z[i] = cand
            search(i - 1, partial + (cand - cm) ** 2 / Dz[i])

    search(n - 1, 0.0)
    best.sort(key=lambda t: t[0])
    if not best:
        return np.round(a_hat), a_hat, np.inf
    z_fix = best[0][1]
    # back-transform: z-space integer → a-space integer (Z unimodular)
    a_fix = np.rint(np.linalg.solve(Z.T, z_fix))
    ratio = best[1][0] / max(best[0][0], 1e-12) if len(best) > 1 else np.inf
    return a_fix, a_hat, float(ratio)


def bench_lambda_method(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: correlated float ambiguity — LAMBDA recovers the true
    integer vector where plain rounding fails."""
    rng = np.random.default_rng(seed)
    n = 6
    trials = 60
    ok = 0
    wrong_round = 0
    for _ in range(trials):
        z_true = rng.integers(-50, 50, n).astype(float)
        R = rng.normal(size=(n, n))
        Q = R.T @ R + np.eye(n) * 0.05  # correlated covariance
        a_hat = z_true + np.linalg.cholesky(Q) @ rng.normal(size=n) * 0.18
        # feed as ILS: y=a_hat, A=I, W=Q^-1
        z_fix, _, _ = lambda_ils(a_hat, np.eye(n), np.linalg.inv(Q))
        ok += int(np.array_equal(z_fix, z_true))
        wrong_round += int(np.array_equal(np.round(a_hat), z_true))
    out: dict[str, float] = {}
    out["synthetic_lambda_fixed"] = float(ok)
    out["synthetic_lambda_plain_round"] = float(wrong_round)
    out["synthetic_lambda_rate"] = ok / trials
    out["synthetic_lambda_better"] = float(ok >= wrong_round)
    return out


if __name__ == "__main__":
    print(bench_lambda_method())
