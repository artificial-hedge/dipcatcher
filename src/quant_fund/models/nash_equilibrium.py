"""Two-player games: fictitious play for zero-sum
matrices (Brown 1951 — converge to minimax), support
enumeration for general bimatrix games, and regret
matching (Hart-Mas-Colell — the CFR workhorse). Synthetic
bench gates fictitious-play exploitability and exact
equilibria on small games."""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def fictitious_play(a: FloatArray, it: int = 2000) -> dict[str, object]:
    """Zero-sum A (row maximizes): each player best-responds
    to the empirical opponent mixture; returns (row mix,
    col mix, game value)."""
    a = np.asarray(a, dtype=np.float64)
    n_r, n_c = a.shape
    rc = np.zeros(n_r)
    cc = np.zeros(n_c)
    for _ in range(it):
        rm = rc / max(rc.sum(), 1)
        cm = cc / max(cc.sum(), 1)
        br_row = int(np.argmax(a @ cm))
        br_col = int(np.argmin(rm @ a))
        rc[br_row] += 1
        cc[br_col] += 1
    row_mix = rc / rc.sum()
    col_mix = cc / cc.sum()
    value = float(row_mix @ a @ col_mix)
    return {"row": row_mix, "col": col_mix, "value": value}


def exploitability(a: FloatArray, row: FloatArray, col: FloatArray) -> float:
    """Zero-sum exploitability = max_i (Ae)_i − min_j (fA)_j
    — 0 at equilibrium."""
    a = np.asarray(a, dtype=np.float64)
    return float((a @ col).max() - (row @ a).min())


def support_enumeration(a: FloatArray, b: FloatArray) -> list[tuple[FloatArray, FloatArray]]:
    """All-equilibria by support enumeration for a
    nondegenerate bimatrix game (A for row, B for col)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n_r, n_c = a.shape
    sols: list[tuple[FloatArray, FloatArray]] = []
    for k in range(1, min(n_r, n_c) + 1):
        for rs in itertools.combinations(range(n_r), k):
            for cs in itertools.combinations(range(n_c), k):
                # col mix makes row indifferent over rs
                bs = np.asarray(b)
                m = bs[np.ix_(list(rs), list(cs))].T
                rhs = np.zeros(k)
                rhs[0] = 1.0
                eq = np.vstack([np.diff(m, axis=0), np.ones(k)])
                try:
                    y = np.linalg.solve(eq, rhs)
                except np.linalg.LinAlgError:
                    continue
                if (y < -1e-9).any() or abs(y.sum() - 1) > 1e-6:
                    continue
                # row mix makes col indifferent over cs
                m2 = np.asarray(a)[np.ix_(list(rs), list(cs))]
                eq2 = np.vstack([np.diff(m2, axis=1).T, np.ones(k)])
                rhs2 = np.zeros(k)
                rhs2[-1] = 1.0
                try:
                    x = np.linalg.solve(eq2, rhs2)
                except np.linalg.LinAlgError:
                    continue
                if (x < -1e-9).any() or abs(x.sum() - 1) > 1e-6:
                    continue
                # no profitable deviation
                xf = np.zeros(n_r)
                xf[list(rs)] = np.clip(x, 0, 1)
                yf = np.zeros(n_c)
                yf[list(cs)] = np.clip(y, 0, 1)
                u_r = a @ yf
                u_c = xf @ b
                if (
                    u_r[list(rs)].min() + 1e-6 >= u_r.max() - 1e-6
                    and u_c[list(cs)].min() + 1e-6 >= u_c.max() - 1e-6
                ):
                    sols.append((xf, yf))
    return sols


def regret_matching(a: FloatArray, it: int = 5000) -> FloatArray:
    """CFR-style regret matching for a zero-sum game played
    by one side; returns the average strategy."""
    a = np.asarray(a, dtype=np.float64)
    n_r, n_c = a.shape
    # opponent col: minimizer — simulate both sides' regrets
    r_pos = np.zeros(n_r)
    c_pos = np.zeros(n_c)
    r_avg = np.zeros(n_r)
    c_avg = np.zeros(n_c)
    rm = np.full(n_r, 1 / n_r)
    cm = np.full(n_c, 1 / n_c)
    for _ in range(it):
        # utilities vs opponent mix
        u_r = a @ cm
        u_c = -(rm @ a)  # col minimizes → payoff = -rm·A
        r_pos += np.maximum(u_r - rm @ u_r, 0)
        c_pos += np.maximum(u_c - cm @ u_c, 0)
        rm = r_pos / r_pos.sum() if r_pos.sum() > 0 else np.full(n_r, 1 / n_r)
        cm = c_pos / c_pos.sum() if c_pos.sum() > 0 else np.full(n_c, 1 / n_c)
        r_avg += rm
        c_avg += cm
    return np.asarray(r_avg / it)


def bench_nash_equilibrium(seed: int = 569) -> dict[str, float]:
    """SYNTHETIC: (a) matching-pennies → FP mixes ≈ uniform;
    (b) random zero-sum → FP exploitability → 0;
    (c) coordination game — support enumeration finds the
    two pure equilibria."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    # matching pennies
    mp = np.array([[1.0, -1.0], [-1.0, 1.0]])
    r = fictitious_play(mp, it=4000)
    row = np.asarray(r["row"])
    out["synthetic_fp_mp_mix_l1"] = float(np.abs(row - 0.5).sum())
    if out["synthetic_fp_mp_mix_l1"] > 0.1:
        raise ValueError(f"fp mp off: {row}")
    # random zero-sum
    a = rng.normal(0, 1, (4, 5))
    r2 = fictitious_play(a, it=20000)
    expl = exploitability(a, np.asarray(r2["row"]), np.asarray(r2["col"]))
    out["synthetic_fp_exploit"] = expl
    if expl > 0.3:
        raise ValueError(f"fp exploit off: {expl}")
    # coordination game: two pure NEs (0,0) and (1,1)
    ag = np.array([[2.0, 0.0], [0.0, 1.0]])
    sols = support_enumeration(ag, ag.copy())
    out["synthetic_ne_sols"] = float(len(sols))
    pure = [s for s in sols if (s[0] > 0.99).sum() == 1 and (s[1] > 0.99).sum() == 1]
    if len(pure) < 2:
        raise ValueError(f"pure NEs missing: {sols}")
    # regret matching on matching pennies
    rm = regret_matching(mp, it=8000)
    out["synthetic_rm_mp_mix_l1"] = float(np.abs(rm - 0.5).sum())
    if out["synthetic_rm_mp_mix_l1"] > 0.15:
        raise ValueError(f"rm mp off: {rm}")
    return out
