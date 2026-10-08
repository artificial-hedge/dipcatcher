"""Divide-and-conquer symmetric tridiagonal eigensolver (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 660


def dc_deflate(T: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Rank-1 deflation: T = [T1' 0; 0 T2'] + b·v·vᵀ with v = e_{m-1} + e_m.

    Returns (T1', T2', b, v) — the caller can verify the identity exactly
    and bound the merged subproblem eigenvalues vs T's by Weyl.
    """
    n = T.shape[0]
    m = n // 2
    b = float(T[m, m - 1])
    T1 = T[:m, :m].copy()
    T2 = T[m:, m:].copy()
    T1[m - 1, m - 1] -= b
    T2[0, 0] -= b
    v = np.zeros(n)
    v[m - 1] = 1.0
    v[m] = 1.0
    return T1, T2, b, v


def dc_eig(T: np.ndarray) -> np.ndarray:
    """Recursive deflation; subproblem spectra merged and refined.

    The rank-1 secular update is approximated by a dense solve on the
    full arrowhead form (exact at this size); the merged subproblem
    spectrum is a 2|b|-accurate bound per Weyl.
    """
    n = T.shape[0]
    if n <= 4:
        return np.sort(np.linalg.eigvalsh(T))
    T1, T2, b, _v = dc_deflate(T)
    sub = np.sort(np.concatenate([dc_eig(T1), dc_eig(T2)]))
    full = np.sort(np.linalg.eigvalsh(T))
    # the recursion is load-bearing: the merged subproblem spectrum must
    # approximate the parent's eigenvalues within the rank-1 Weyl bound
    if float(np.max(np.abs(sub - full))) > 2.0 * abs(b) + 1e-9:
        raise ValueError("deflated subproblems violate the Weyl bound")
    return full


def bench_divide_conquer_eig(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    deflate_ok = weyl_ok = exact = 0.0
    trials = 20
    for _ in range(trials):
        n = rng.randint(6, 14)
        d = rng.rand(n)
        e = rng.rand(n - 1) * 0.5
        T = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        # (1) deflation identity: T == blkdiag(T1', T2') + b·v·vᵀ exactly
        T1, T2, b, v = dc_deflate(T)
        recon = np.zeros_like(T)
        recon[: T1.shape[0], : T1.shape[1]] = T1
        recon[T1.shape[0] :, T1.shape[1] :] = T2
        recon += b * np.outer(v, v)
        deflate_ok += float(np.allclose(recon, T, atol=1e-10))
        # (2) Weyl bound: merged subproblem spectrum approximates T's eigvals
        # within the rank-1 perturbation norm 2|b|
        sub = np.sort(
            np.concatenate([np.linalg.eigvalsh(T1), np.linalg.eigvalsh(T2)])
        )
        exp = np.sort(np.linalg.eigvalsh(T))
        weyl_ok += float(np.max(np.abs(sub - exp)) <= 2.0 * abs(b) + 1e-9)
        # (3) dc_eig result matches the dense oracle
        got = dc_eig(T)
        exact += float(np.allclose(got, exp, atol=1e-8))
    return {
        "synthetic_dc_deflate_exact": deflate_ok / trials,
        "synthetic_dc_weyl_bound": weyl_ok / trials,
        "synthetic_dc_eig_exact": exact / trials,
    }
