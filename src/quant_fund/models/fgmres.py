"""FGMRES: flexible GMRES with variable preconditioner (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 665


def fgmres(A: np.ndarray, b: np.ndarray, M_inv: np.ndarray, iters: int) -> np.ndarray:
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x
    beta = np.linalg.norm(r)
    V = [r / beta]
    Z = []
    H = np.zeros((iters + 1, iters))
    for j in range(iters):
        z = M_inv @ V[j]
        Z.append(z)
        w = A @ z
        for i in range(j + 1):
            H[i, j] = w @ V[i]
            w = w - H[i, j] * V[i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] > 1e-12:
            V.append(w / H[j + 1, j])
        # solve least squares
        e1 = np.zeros(j + 2)
        e1[0] = beta
        y = np.linalg.lstsq(H[: j + 2, : j + 1], e1, rcond=None)[0]
    x = np.stack(Z, axis=1) @ y
    return x


def bench_fgmres(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 15
    for _ in range(trials):
        n = 15
        A = rng.rand(n, n) + n * np.eye(n)
        b = rng.rand(n)
        M_inv = np.diag(1.0 / np.diag(A))
        x = fgmres(A, b, M_inv, 12)
        ok += float(np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 0.1)
    return {"synthetic_fgmres_conv": ok / trials}
