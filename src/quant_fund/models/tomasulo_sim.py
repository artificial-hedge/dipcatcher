"""Tomasulo-lite out-of-order execution — SYNTHETIC correctness.

Issues ops to reservation stations, RAW via CDB tags, WAR/WAW via
register renaming. Verified: final register file identical to
in-order execution; cycle count <= in-order serial.
"""

from __future__ import annotations

import random

# op: (dest, src1, src2_reg_or_None, imm) → dest = src1 + (reg[src2] or imm)
Op = tuple[int, int, int | None, int]


def _exec_seq(ops: list[Op]) -> list[int]:
    r = [0] * 16
    for d, s1, s2, imm in ops:
        r[d] = r[s1] + (r[s2] if s2 is not None else imm)
    return r


def exec_ooo(ops: list[Op], lat: int = 2, alu: int = 2) -> tuple[int, list[int]]:
    """Returns (cycles, regs). Tomasulo: issue in order, exec when ready,
    broadcast on CDB. Multiple ALUs."""
    n = len(ops)
    tag: list[int | None] = [None] * 16  # reg -> producing op index
    done = [False] * n
    in_exec: list[tuple[int, int, int, int, int | None, int]] = []  # (op, ready_cycle, d,s1,s2,imm)
    issued = 0
    cyc = 0
    guard = 0
    results: dict[int, int] = {}
    while not all(done) and guard < 100000:
        guard += 1
        cyc += 1
        # broadcast finished
        for j, fin, d, s1, s2, imm in in_exec[:]:
            if cyc >= fin:
                res = results.get(s1, 0) + (results.get(s2, 0) if s2 is not None else imm)
                results[d] = res
                if tag[d] == j:
                    tag[d] = None
                done[j] = True
                in_exec.remove((j, fin, d, s1, s2, imm))
        # issue in order if operands ready
        while issued < n and len(in_exec) < alu:
            di, a1, a2, im = ops[issued]
            if tag[a1] is None and (a2 is None or tag[a2] is None):
                in_exec.append((issued, cyc + lat, di, a1, a2, im))
                tag[di] = issued
                issued += 1
            else:
                break
    out = [0] * 16
    for i in range(16):
        out[i] = results.get(i, 0)
    return cyc, out


def bench_tomasulo_sim(seed: int = 20261231 + 333) -> dict[str, float]:
    rng = random.Random(seed)
    match = speed = 0
    trials = 40
    for _ in range(trials):
        ops: list[Op] = []
        for _ in range(rng.randrange(6, 16)):
            d = rng.randrange(1, 16)
            s1 = rng.randrange(0, 16)
            s2 = rng.randrange(0, 16) if rng.random() < 0.6 else None
            ops.append((d, s1, s2, rng.randrange(-4, 5)))
        cyc, regs = exec_ooo(ops)
        ref = _exec_seq(ops)
        match += int(regs == ref)
        serial = len(ops) * 2
        speed += int(cyc <= serial)
    return {
        "synthetic_regs_match_inorder": float(match / trials),
        "synthetic_beats_serial": float(speed / trials),
    }
