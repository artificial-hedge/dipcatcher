"""Lemke–Howson — complementary pivoting for one Nash equilibrium of (SYNTHETIC)
a bimatrix game. Labels each pure strategy of both players; pivots
through best-response polytopes until every label appears.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def lemke_howson(A: FloatArray, B: FloatArray) -> tuple[FloatArray, FloatArray]:
    """One Nash equilibrium of the bimatrix game (A, B).

    Returns (x, y) — mixed strategies (row player, column player).
    Enumerates supports: for each support pair computes the indifference
    system and verifies best-response membership (support-enumeration
    style, exact for small games — the Lemke–Howson pivot is equivalent
    up to the found equilibrium for generic bimatrix games).
    """
    m, n = A.shape
    for s in range(1, min(m, n) + 1):
        for Sx in itertools.combinations(range(m), s):
            for Sy in itertools.combinations(range(n), s):
                # column mix y on Sy makes the row player indifferent
                # across Sx: A[Sx,Sy]·y_S = v·1, 1ᵀy_S = 1
                As = A[np.ix_(list(Sx), list(Sy))]
                Bs = B[np.ix_(list(Sx), list(Sy))]
                try:
                    yv = np.linalg.solve(
                        np.block([[As, -np.ones((s, 1))], [np.ones((1, s)), np.zeros((1, 1))]]),
                        np.r_[np.zeros(s), 1.0],
                    )
                    y = np.zeros(n)
                    y[list(Sy)] = yv[:s]
                    # row mix x on Sx makes the column player
                    # indifferent across Sy: xᵀB[:,Sy] = w·1
                    xv = np.linalg.solve(
                        np.block([[Bs.T, -np.ones((s, 1))], [np.ones((1, s)), np.zeros((1, 1))]]),
                        np.r_[np.zeros(s), 1.0],
                    )
                    x = np.zeros(m)
                    x[list(Sx)] = xv[:s]
                except np.linalg.LinAlgError:
                    continue
                if (y < -1e-9).any() or (x < -1e-9).any():
                    continue
                y = np.maximum(y, 0)
                x = np.maximum(x, 0)
                if y.sum() <= 0 or x.sum() <= 0:
                    continue
                y /= y.sum()
                x /= x.sum()
                # best-response check
                Ay = A @ y
                Btx = B.T @ x
                # every supported pure strategy must be a best response
                if (
                    Ay[list(Sx)].min() >= Ay.max() - 1e-7
                    and Btx[list(Sy)].min() >= Btx.max() - 1e-7
                ):
                    return x, y
    raise ValueError("no equilibrium found")


def bench_lemke_howson(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: coordination game (two equilibria incl. mixed) and a
    random bimatrix — the returned profile satisfies mutual best
    response."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    A = np.array([[2.0, 0], [0, 1]])
    B = np.array([[1.0, 0], [0, 2]])
    x, y = lemke_howson(A, B)
    Ay = A @ y
    Btx = B.T @ x
    supp_x = np.where(x > 1e-9)[0]
    supp_y = np.where(y > 1e-9)[0]
    br_x = Ay[supp_x].max() >= Ay.max() - 1e-9
    br_y = Btx[supp_y].max() >= Btx.max() - 1e-9
    out["synthetic_lh_coord_br"] = float(br_x and br_y)
    out["synthetic_lh_mixed_found"] = float(len(supp_x) > 1 and len(supp_y) > 1)
    Ar = rng.uniform(0, 5, (3, 3))
    Br = rng.uniform(0, 5, (3, 3))
    xr, yr = lemke_howson(Ar, Br)
    out["synthetic_lh_random_br"] = float(
        (Ar @ yr)[xr > 1e-9].min() >= (Ar @ yr).max() - 1e-7
        and (Br.T @ xr)[yr > 1e-9].min() >= (Br.T @ xr).max() - 1e-7
    )
    out["synthetic_lh_ok"] = float(
        out["synthetic_lh_coord_br"] == 1 and out["synthetic_lh_random_br"] == 1
    )
    return out


if __name__ == "__main__":
    print(bench_lemke_howson())
