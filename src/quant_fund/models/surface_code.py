"""Rotated distance-3 surface code [[9,1,3]] with exact ML decoding (SYNTHETIC).

Nine data qubits on a 3x3 grid; stabilizers are the rotated-code
checkerboard: weight-4 interior plaquettes plus weight-2 boundary halves.
X errors are decoded from the 4-bit Z-plaquette syndrome (and Z errors from
the X-star syndrome symmetrically). The decoder enumerates all 512 error
patterns and keeps a minimum-weight representative per syndrome — exact
maximum-likelihood, no MWPM approximation needed at this size. Logical
failure is scored modulo the stabilizer group: the correction fails iff
e + decode(syndrome(e)) is a nontrivial logical operator.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 946

# data qubit index = 3*row + col on the 3x3 grid
X_STABS = np.array(
    [
        [1, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
    ],
    dtype=np.uint8,
)
Z_STABS = np.array(
    [
        [0, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1],
    ],
    dtype=np.uint8,
)


def _stab_row_space(stabs: np.ndarray) -> set[int]:
    n = stabs.shape[0]
    out = set()
    for comb in range(1 << n):
        v = np.zeros(9, dtype=np.uint8)
        for i in range(n):
            if (comb >> i) & 1:
                v ^= stabs[i]
        out.add(int(np.packbits(v)[0]) if False else _vecint(v))
    return out


def _vecint(v: np.ndarray) -> int:
    return int(sum(int(b) << i for i, b in enumerate(v)))


def _intvec(x: int) -> np.ndarray:
    return np.array([(x >> i) & 1 for i in range(9)], dtype=np.uint8)


def build_decode_table(
    synd_stabs: np.ndarray,
) -> dict[int, int]:
    """syndrome -> minimum-weight error pattern (as int bitmask)."""
    table: dict[int, int] = {}
    for e in range(512):
        ev = _intvec(e)
        syn = _vecint((synd_stabs @ ev) % 2)
        if syn not in table or bin(e).count("1") < bin(table[syn]).count("1"):
            table[syn] = e
    return table


def logical_failure(
    e: int, correction: int, synd_stabs: np.ndarray, comp_stabs: np.ndarray
) -> bool:
    """True iff e+correction is a nontrivial logical operator:
    it commutes with all syndrome checks (in ker) but is not itself a
    stabilizer product (not in rowspan of the complementary checks)."""
    res = _intvec(e) ^ _intvec(correction)
    if not res.any():
        return False
    if ((synd_stabs @ res) % 2).any():
        return True  # should not happen for a valid decoder
    # residual is in ker(syndrome checks); stabilizer iff in rowspan of the
    # complementary-type stabilizers
    return _vecint(res) not in _stab_row_space(comp_stabs)


def bench_surface_code(seed: int = _SEED) -> dict[str, float]:
    _ = np.random.default_rng(seed)
    checks = []
    # code sanity: commuting stabilizers, distance 3
    checks.append(not (X_STABS @ Z_STABS.T % 2).any())
    zrow = _stab_row_space(Z_STABS)
    xrow = _stab_row_space(X_STABS)
    ker_z = [v for v in range(512) if not ((Z_STABS @ _intvec(v)) % 2).any()]
    dz = min(bin(v).count("1") for v in ker_z if v not in xrow)
    checks.append(dz == 3)
    table = build_decode_table(Z_STABS)
    # every correctable single error decodes exactly
    single_ok = all(
        not logical_failure(
            1 << q,
            table[_vecint((Z_STABS @ _intvec(1 << q)) % 2)],
            Z_STABS,
            X_STABS,
        )
        for q in range(9)
    )
    checks.append(single_ok)

    def logical_rate(p: float) -> float:
        """Exact logical failure rate: binomial-weighted sum over all 512
        error patterns of the decoder's miscorrection indicator."""
        tot = 0.0
        for e in range(512):
            w = bin(e).count("1")
            prob = p**w * (1 - p) ** (9 - w)
            syn = _vecint((Z_STABS @ _intvec(e)) % 2)
            tot += prob * logical_failure(e, table[syn], Z_STABS, X_STABS)
        return tot

    # exact rates: the code must beat the unprotected single-qubit rate p
    # at low p (distance-3 ML decoder scales ~ p^2), and improve with p->0
    r02, r04 = logical_rate(0.02), logical_rate(0.04)
    checks.append(r02 < 0.02)
    checks.append(r04 < 0.04)
    checks.append(r02 < r04)
    del zrow
    return {"synthetic_surface_code": float(np.mean(checks))}
