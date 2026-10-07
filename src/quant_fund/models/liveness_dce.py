"""Backward liveness analysis + dead-code elimination (synthetic) (SYNTHETIC).

Straight-line instruction list: (dst, src1, src2). live-out via
use/def dataflow; DCE removes assignments to never-live vars.
Verified: (i) live sets agree with brute-force "needed" oracle;
(ii) eliminated program computes identical outputs; (iii) DCE is
idempotent.
"""

from __future__ import annotations

import random

Insn = tuple[str, str, str]  # dst = src1 OP src2


def liveness(insns: list[Insn], live_out: set[str]) -> list[set[str]]:
    n = len(insns)
    live: list[set[str]] = [set() for _ in range(n + 1)]
    live[n] = set(live_out)
    for i in range(n - 1, -1, -1):
        d, s1, s2 = insns[i]
        live[i] = (live[i + 1] - {d}) | {s1, s2}
    return live


def dce(insns: list[Insn], live_out: set[str]) -> list[Insn]:
    """Iteratively drop dst-dead assignments until fixpoint."""
    cur = insns[:]
    while True:
        live = liveness(cur, live_out)
        new = [ins for i, ins in enumerate(cur) if ins[0] in live[i + 1]]
        if len(new) == len(cur):
            return cur
        cur = new


def _exec(insns: list[Insn], vals: dict[str, int]) -> dict[str, int]:
    env = dict(vals)
    for d, s1, s2 in insns:
        env[d] = env.get(s1, 0) + env.get(s2, 0)
    return env


def bench_liveness_dce(seed: int = 20261231 + 284) -> dict[str, float]:
    rng = random.Random(seed)
    names = [f"v{i}" for i in range(6)]
    agree = same_out = idem = 0
    trials = 40
    for _ in range(trials):
        n = rng.randint(4, 10)
        insns = [(rng.choice(names), rng.choice(names), rng.choice(names)) for _ in range(n)]
        out_vars = set(rng.sample(names, rng.randint(1, 3)))
        live = liveness(insns, out_vars)
        # oracle: v live at point i iff a later use of v occurs before any
        # redefinition of v (or v ∈ live_out at the end)
        ok = True
        for i in range(n):
            for v in names:
                oracle = v in out_vars and all(insns[j][0] != v for j in range(i, n))
                for j in range(i, n):
                    d, s1, s2 = insns[j]
                    if v in (s1, s2):
                        oracle = True
                        break
                    if d == v:
                        break
                ok = ok and (v in live[i]) == oracle
        agree += int(ok)
        vals = {v: rng.randint(0, 9) for v in names}
        e1 = _exec(insns, vals)
        e2 = _exec(dce(insns, out_vars), vals)
        same_out += int(all(e1[v] == e2[v] for v in out_vars))
        d1 = dce(insns, out_vars)
        d2 = dce(d1, out_vars)
        idem += int(d1 == d2)
    return {
        "synthetic_live_agree": float(agree / trials),
        "synthetic_same_output": float(same_out / trials),
        "synthetic_idempotent": float(idem / trials),
    }
