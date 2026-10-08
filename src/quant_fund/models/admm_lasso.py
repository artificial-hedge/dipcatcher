"""ADMM for lasso: min ½‖Ax−b‖² + λ‖x‖₁ (x-update + soft-threshold split) (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 573


def _soft(v: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)


def admm_lasso(
    A: np.ndarray, b: np.ndarray, lam: float, rho: float = 1.0, iters: int = 400
) -> np.ndarray:
    n = A.shape[1]
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)
    M = A.T @ A + rho * np.eye(n)
    Minv = np.linalg.solve(M, np.eye(n))
    for _ in range(iters):
        x = Minv @ (A.T @ b + rho * (z - u))
        z = _soft(x + u, lam / rho)
        u = u + x - z
    return z


def _obj(A: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> float:
    return float(0.5 * np.sum((A @ x - b) ** 2) + lam * np.sum(np.abs(x)))


def _coord_oracle(A: np.ndarray, b: np.ndarray, lam: float, iters: int = 2000) -> np.ndarray:
    n = A.shape[1]
    x = np.zeros(n)
    col2 = (A**2).sum(0)
    for _ in range(iters):
        for j in range(n):
            r = b - A @ x + A[:, j] * x[j]
            rho_j = A[:, j] @ r
            x[j] = np.sign(rho_j) * max(abs(rho_j) - lam, 0) / col2[j]
    return x


def bench_admm_lasso(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 30
    ok = dec = 0
    for _ in range(n):
        m_, n_ = 8, 5
        A = rng.uniform(-1, 1, (m_, n_))
        b = rng.uniform(-1, 1, m_)
        lam = rng.uniform(0.05, 1.0)
        x = admm_lasso(A, b, lam)
        ref = _coord_oracle(A, b, lam)
        ok += int(np.isclose(_obj(A, b, x, lam), _obj(A, b, ref, lam), atol=2e-3))
        dec += int(_obj(A, b, x, lam) <= _obj(A, b, np.zeros(n_), lam))
    return {
        "synthetic_admm_optimal": float(ok / n),
        "synthetic_beats_zero": float(dec / n),
    }
