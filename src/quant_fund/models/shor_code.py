"""Shor [[9,1,3]] code: concatenated 3-qubit repetition in X and Z bases (SYNTHETIC).

|0_L> = ((|000>+|111>)/sqrt2)^⊗3, |1_L> = ((|000>-|111>)/sqrt2)^⊗3.
Decode: per-block majority vote repairs bit flips coherently (amplitudes at
error patterns map back onto the majority pattern and add), then a majority
vote across the three block signs repairs a single phase flip. Corrects any
single-qubit Pauli error; verified on 2^9 statevectors.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 947


def _idx(bits: list[int] | tuple[int, ...]) -> int:
    return sum(b << (8 - q) for q, b in enumerate(bits))


def encode_zero() -> np.ndarray:
    """|0_L> statevector over 9 qubits."""
    psi = np.zeros(512, dtype=np.complex128)
    for blk in range(8):
        bits = [(blk >> (2 - b)) & 1 for b in range(3)]
        psi[_idx([bits[0]] * 3 + [bits[1]] * 3 + [bits[2]] * 3)] = 1.0 / (2 * np.sqrt(2))
    return psi


def apply_pauli(psi: np.ndarray, q: int, pauli: str) -> np.ndarray:
    """Apply Pauli ('X','Y','Z') on qubit q (q=0 is the index MSB)."""
    out = np.zeros_like(psi)
    mask = 1 << (8 - q)
    for i in range(512):
        if not i & mask:
            j = i | mask
            if pauli == "X":
                out[j] += psi[i]
                out[i] += psi[j]
            elif pauli == "Y":
                out[j] += 1j * psi[i]
                out[i] += -1j * psi[j]
            else:
                out[i] += psi[i]
                out[j] += -psi[j]
    return out


def _bits_of(idx: int) -> list[int]:
    return [(idx >> (8 - q)) & 1 for q in range(9)]


def decode_state(psi: np.ndarray) -> np.ndarray:
    """Project an errored code state back onto the code space."""
    # step 1: bit flips — coherent per-block majority vote
    bit_fixed = np.zeros_like(psi)
    for i in range(512):
        if psi[i] == 0:
            continue
        bits = _bits_of(i)
        new: list[int] = []
        for b in range(3):
            seg = bits[3 * b : 3 * b + 3]
            maj = 1 if sum(seg) >= 2 else 0
            new += [maj] * 3
        bit_fixed[_idx(new)] += psi[i]
    # step 2: phase flips — block sign relative to pattern 000
    ref = _idx([0] * 9)
    eps = 1e-9
    signs = []
    for b in range(3):
        e = [0] * 9
        for qb in range(3 * b, 3 * b + 3):
            e[qb] = 1
        a0 = bit_fixed[ref]
        a1 = bit_fixed[_idx(e)]
        if abs(a0) < eps or abs(a1) < eps:
            signs.append(0)
            continue
        signs.append(1 if np.real(a1 / a0) >= 0 else -1)
    nz = [s for s in signs if s != 0]
    target = 1 if sum(nz) >= 0 else -1
    out = bit_fixed.copy()
    for b, s in enumerate(signs):
        if s != 0 and s != target:
            for i in range(512):
                if _bits_of(i)[3 * b]:  # block b qubits are uniform
                    out[i] = -out[i]
    return out


def bench_shor_code(seed: int = _SEED) -> dict[str, float]:
    _ = np.random.default_rng(seed)
    psi0 = encode_zero()
    ok = 0
    trials = 0
    for q in range(9):
        for pauli in ("X", "Y", "Z"):
            err = apply_pauli(psi0, q, pauli)
            rec = decode_state(err)
            rec = rec / np.linalg.norm(rec)
            ok += abs(np.vdot(psi0, rec)) ** 2 > 0.999
            trials += 1
    rec = decode_state(psi0)
    ok += abs(np.vdot(psi0, rec / np.linalg.norm(rec))) ** 2 > 0.999
    trials += 1
    # weight-2 phase error across two blocks is a logical error the decoder
    # must NOT conceal: fidelity to |0_L> stays low (honest negative)
    err2 = apply_pauli(apply_pauli(psi0, 0, "Z"), 3, "Z")
    rec2 = decode_state(err2)
    rec2 = rec2 / np.linalg.norm(rec2)
    ok += abs(np.vdot(psi0, rec2)) ** 2 < 0.5
    trials += 1
    return {"synthetic_shor_code": ok / trials}
