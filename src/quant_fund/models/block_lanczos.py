"""Block Lanczos: B-step Krylov basis for symmetric A."""

import numpy as np

_SEED = 20261231 + 662


def block_lanczos(A: np.ndarray, b: int, steps: int, rng: np.random.RandomState) -> np.ndarray:
    n = A.shape[0]
    Q0, _ = np.linalg.qr(rng.rand(n, b))
    blocks = [Q0]
    alpha = []
    beta_prev = np.zeros((b, b))
    Q_prev = np.zeros((n, b))
    Q_cur = Q0
    for _ in range(steps):
        Z = A @ Q_cur - Q_prev @ beta_prev.T
        a = Q_cur.T @ Z
        alpha.append(a)
        R = Z - Q_cur @ a
        Q_next, b_next = np.linalg.qr(R)
        blocks.append(Q_next)
        beta_prev = b_next
        Q_prev, Q_cur = Q_cur, Q_next
        if np.linalg.norm(b_next) < 1e-10:
            break
    # Ritz values from block tridiagonal
    k = len(alpha)
    T = np.zeros((k * b, k * b))
    for i, a in enumerate(alpha):
        T[i * b : (i + 1) * b, i * b : (i + 1) * b] = a
        if i + 1 < k:
            T[i * b : (i + 1) * b, (i + 1) * b : (i + 2) * b] = np.eye(b)
            T[(i + 1) * b : (i + 2) * b, i * b : (i + 1) * b] = np.eye(b)
    return np.linalg.eigvalsh(T)


def bench_block_lanczos(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 15
    for _ in range(trials):
        n = 20
        A = rng.rand(n, n)
        A = A @ A.T
        # extremes of the spectrum should be approximated
        ritz = block_lanczos(A, 4, 5, rng)
        tru = np.linalg.eigvalsh(A)
        ok += float(abs(ritz.max() - tru.max()) / tru.max() < 0.3)
    return {"synthetic_block_lanczos_top": ok / trials}
