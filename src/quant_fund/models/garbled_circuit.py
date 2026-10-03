"""Yao garbled-circuit evaluation for small Boolean circuits (SYNTHETIC bench)."""

from __future__ import annotations

import hashlib
import random


def _k(seed: int, label: int) -> bytes:
    return hashlib.sha256(f"{seed}:{label}".encode()).digest()[:16]


def garble(
    gate: tuple[str, int, int, int], keys: dict[int, tuple[int, int]], seed: int
) -> list[bytes]:
    """Garble gate (op, in1, in2, out): 4 ciphertexts rows[a][b]."""
    op, i1, i2, _ = gate
    table = {
        "and": [[0, 0], [0, 1]],
        "or": [[0, 1], [1, 1]],
        "xor": [[0, 1], [1, 0]],
        "nand": [[1, 1], [1, 0]],
    }[op]
    k_in = [keys[i1], keys[i2]]
    k_out = keys[gate[3]]
    ct = []
    for a in (0, 1):
        for b in (0, 1):
            out_label = k_out[table[a][b]]
            msg = _k(seed, k_in[0][a] * 2 + k_in[1][b])
            ct.append(
                bytes(x ^ y for x, y in zip(msg[:2], out_label.to_bytes(2, "big"), strict=True))
            )
    return ct


def evaluate(
    cts: list[bytes],
    in_keys: tuple[int, int],
    out_keymap: dict[int, int],
    gate: tuple[str, int, int, int],
    seed: int,
) -> int:
    """Try all 4 rows; the row decryptable with our two input keys yields the output label."""
    for a, b, ct in zip((0, 0, 1, 1), (0, 1, 0, 1), cts, strict=True):
        msg = _k(seed, in_keys[0] * 2 + in_keys[1])
        if a * 0 + b * 0 == 0:
            dec = bytes(x ^ y for x, y in zip(msg[:2], ct, strict=True))
            v = int.from_bytes(dec, "big")
            if v in out_keymap.values():
                return v
    raise ValueError("no row decrypted")


def garbled_eval(
    circuit: list[tuple[str, int, int, int]], inputs: dict[int, int], seed: int = 0
) -> int:
    """Full Yao-lite: garble each gate, evaluator follows the label path."""
    rng = random.Random(seed)
    wire_keys = {}
    for gate in circuit:
        for w in (gate[1], gate[2], gate[3]):
            if w not in wire_keys:
                wire_keys[w] = (rng.getrandbits(14), rng.getrandbits(14))
    for w in inputs:
        if w not in wire_keys:
            wire_keys[w] = (rng.getrandbits(14), rng.getrandbits(14))
    out_wire = circuit[-1][3]
    result_label = None
    wire_labels: dict[int, int] = {}
    for gate in circuit:
        cts = garble(gate, wire_keys, seed)

        def _key(w: int) -> int:
            if w in inputs:
                return wire_keys[w][inputs[w]]
            if w in wire_labels:
                return wire_labels[w]
            raise ValueError("gate inputs not ready")

        in_keys = (_key(gate[1]), _key(gate[2]))
        label = evaluate(cts, in_keys, dict(enumerate(wire_keys[gate[3]])), gate, seed)
        wire_labels[gate[3]] = label
        result_label = label
    return wire_keys[out_wire].index(result_label)


def _bench_garbled_circuit(seed: int = 0) -> float:
    checks = []
    # single AND gate: wires 0,1 -> 2
    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
        checks.append(garbled_eval([("and", 0, 1, 2)], {0: a, 1: b}, seed) == (a & b))
    # xor then and: (a xor b) and c
    circ = [("xor", 0, 1, 3), ("and", 3, 2, 4)]
    for a, b, c in ((0, 0, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1)):
        checks.append(garbled_eval(circ, {0: a, 1: b, 2: c}, seed) == ((a ^ b) & c))
    return sum(checks) / len(checks)


def bench_garbled_circuit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_garbled_circuit": _bench_garbled_circuit(seed)}
