"""5-point Jacobi stencil on a decomposed grid with halo exchange."""

import numpy as np

_SEED = 20261231 + 640


def _decomp_stencil(g: np.ndarray, n_procs: int, iters: int) -> np.ndarray:
    n = g.shape[0]
    rows = np.array_split(np.arange(n), n_procs)
    local = [g[r].copy() for r in rows]
    for _ in range(iters):
        # halo exchange
        ghost_up: list[np.ndarray | None] = [
            local[i - 1][-1] if i > 0 else None for i in range(n_procs)
        ]
        ghost_dn: list[np.ndarray | None] = [
            local[i + 1][0] if i < n_procs - 1 else None for i in range(n_procs)
        ]
        new_local = []
        for k, blk in enumerate(local):
            gu = ghost_up[k]
            gd = ghost_dn[k]
            up = (
                np.vstack([gu[None, :], blk[:-1]])
                if gu is not None
                else np.vstack([blk[:1], blk[:-1]])
            )
            dn = (
                np.vstack([blk[1:], gd[None, :]])
                if gd is not None
                else np.vstack([blk[1:], blk[-1:]])
            )
            left = np.hstack([blk[:, :1], blk[:, :-1]])
            right = np.hstack([blk[:, 1:], blk[:, -1:]])
            new_local.append(0.25 * (up + dn + left + right))
        local = new_local
    return np.vstack(local)


def bench_stencil_halo(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    errs = []
    for _ in range(6):
        n = 40
        g = rng.rand(n, n)
        it = 15
        par = _decomp_stencil(g, 4, it)
        # serial oracle
        ref = g.copy()
        for _ in range(it):
            ref = 0.25 * (
                np.vstack([ref[:1], ref[:-1]])
                + np.vstack([ref[1:], ref[-1:]])
                + np.hstack([ref[:, :1], ref[:, :-1]])
                + np.hstack([ref[:, 1:], ref[:, -1:]])
            )
        errs.append(float(np.abs(par - ref).max()))
    return {
        "synthetic_halo_max_err": float(np.mean(errs)),
        "synthetic_halo_correct": float(np.mean(errs) < 1e-10),
    }
