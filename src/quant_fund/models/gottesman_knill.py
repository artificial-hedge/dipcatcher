"""Gottesman-Knill stabilizer-tableau simulator (SYNTHETIC).

n-qubit state tracked as a 2n x (2n+1) binary tableau: rows 0..n-1
destabilizers, n..2n-1 stabilizers; columns 0..n-1 X bits, n..2n-1 Z bits,
column 2n the phase bit. Supports H, S, CNOT, Pauli gates, and Z measurement
with the Aaronson-Gottesman deterministic/random outcome rules.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 944


def _g(a1: int, b1: int, a2: int, b2: int) -> int:
    """Phase correction for Pauli multiplication: g in {0,1,-1}."""
    if a1 == 0 and b1 == 0:
        return 0
    if a1 == 1 and b1 == 0:  # X * P
        return b2 * (2 * a2 - 1)
    if a1 == 0 and b1 == 1:  # Z * P
        return a2 * (1 - 2 * b2)
    return b2 - a2  # Y * P


def _rowsum_into(dst: np.ndarray, src: np.ndarray, n: int) -> None:
    """dst := dst * src as Pauli products (binary symplectic + phase)."""
    x1, z1 = src[:n], src[n : 2 * n]
    x2, z2 = dst[:n], dst[n : 2 * n]
    ph = 2 * int(dst[2 * n]) + 2 * int(src[2 * n])
    ph += sum(_g(int(x1[k]), int(z1[k]), int(x2[k]), int(z2[k])) for k in range(n))
    dst[: 2 * n] ^= src[: 2 * n]
    dst[2 * n] = (ph % 4) // 2


def _rowsum(tab: np.ndarray, h: int, i: int, n: int) -> None:
    _rowsum_into(tab[h], tab[i], n)


def h(tab: np.ndarray, q: int, n: int) -> None:
    tab[:, 2 * n] ^= tab[:, q] & tab[:, n + q]
    tab[:, [q, n + q]] = tab[:, [n + q, q]]


def s_gate(tab: np.ndarray, q: int, n: int) -> None:
    tab[:, 2 * n] ^= tab[:, q] & tab[:, n + q]
    tab[:, n + q] ^= tab[:, q]


def cnot(tab: np.ndarray, c: int, t: int, n: int) -> None:
    xc = tab[:, c].astype(bool)
    zc = tab[:, n + c].astype(bool)
    xt = tab[:, t].astype(bool)
    zt = tab[:, n + t].astype(bool)
    phase = xc & zt & ~(xt ^ zc)
    tab[:, 2 * n] ^= phase.astype(np.uint8)
    tab[:, t] ^= tab[:, c]
    tab[:, n + c] ^= tab[:, n + t]


def x(tab: np.ndarray, q: int, n: int) -> None:
    tab[:, 2 * n] ^= tab[:, n + q]


def z(tab: np.ndarray, q: int, n: int) -> None:
    tab[:, 2 * n] ^= tab[:, q]


def init_state(n: int) -> np.ndarray:
    """Tableau for |0...0>: destabilizers X_i, stabilizers Z_i."""
    tab = np.zeros((2 * n, 2 * n + 1), dtype=np.uint8)
    for i in range(n):
        tab[i, i] = 1  # destabilizer X_i
        tab[n + i, n + i] = 1  # stabilizer Z_i
    return tab


def measure(tab: np.ndarray, q: int, n: int, rng: np.random.Generator) -> int:
    """Measure Z_q, collapsing the tableau. Returns outcome bit."""
    stab_x = tab[n:, q]
    if stab_x.any():
        # random: some stabilizer anticommutes with Z_q
        p = int(np.flatnonzero(stab_x)[0]) + n
        for i in range(2 * n):
            if i != p and tab[i, q]:
                _rowsum(tab, i, p, n)
        tab[p - n] = tab[p].copy()
        tab[p] = 0
        tab[p, n + q] = 1
        out = int(rng.integers(0, 2))
        tab[p, 2 * n] = out
        return out
    # deterministic: +-Z_q = prod of stabilizer rows S_i over
    # A = {i : destabilizer row i has X bit on q}
    scratch = np.zeros(2 * n + 1, dtype=np.uint8)
    for i in range(n):
        if tab[i, q]:
            _rowsum_into(scratch, tab[n + i], n)
    return int(scratch[2 * n])


def stabilizer_vec(tab: np.ndarray, k: int, n: int) -> str:
    """Pauli string for stabilizer row n+k (ignores sign)."""
    row = tab[n + k]
    out = []
    for q in range(n):
        xx, zz = int(row[q]), int(row[n + q])
        if not xx and not zz:
            out.append("I")
        elif xx and not zz:
            out.append("X")
        elif zz and not xx:
            out.append("Z")
        else:
            out.append("Y")
    return "".join(out)


def bench_gottesman_knill(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    checks = []
    n = 2
    # Bell state: stabilizers {XX, ZZ}; ZZ outcomes correlated.
    tab = init_state(n)
    h(tab, 0, n)
    cnot(tab, 0, 1, n)
    stabs = {stabilizer_vec(tab, k, n) for k in range(n)}
    checks.append("XX" in stabs and "ZZ" in stabs)
    a = measure(tab, 0, n, rng)
    b = measure(tab, 1, n, rng)
    checks.append(a == b)
    # |0> measured in Z is deterministically 0.
    checks.append(measure(init_state(1), 0, 1, rng) == 0)
    # X|0> = |1>: deterministic 1.
    tab4 = init_state(1)
    x(tab4, 0, 1)
    checks.append(measure(tab4, 0, 1, rng) == 1)
    # GHZ on 3 qubits: all outcomes equal.
    tab5 = init_state(3)
    h(tab5, 0, 3)
    cnot(tab5, 0, 1, 3)
    cnot(tab5, 0, 2, 3)
    outs = [measure(tab5, q, 3, rng) for q in range(3)]
    checks.append(outs[0] == outs[1] == outs[2])
    # S gate leaves |0> unchanged.
    tab6 = init_state(1)
    s_gate(tab6, 0, 1)
    checks.append(measure(tab6, 0, 1, rng) == 0)
    return {"synthetic_gottesman_knill": float(np.mean(checks))}
