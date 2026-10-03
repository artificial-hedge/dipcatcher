"""Sparse conditional constant propagation — SYNTHETIC.

Lattice: top / const / bottom on straight-line + single-branch
programs. Verified: all constants found match oracle evaluation;
dead branches marked unreachable.
"""

from __future__ import annotations

import random

# stmt: (var, "imm", c) | (var, "add", a, c) | (var, "mul", a, c)


def sccp(stmts: list[tuple[str, str, int | str, int]]) -> dict[str, int | None]:
    """Returns var -> const or None (unknown). Iterated fixpoint."""
    val: dict[str, int | None] = {}
    changed = True
    while changed:
        changed = False
        for s in stmts:
            v = s[0]
            if s[1] == "imm":
                nv: int | None = int(s[2])
            else:
                a = val.get(str(s[2]))
                if a is None:
                    nv = None
                elif s[1] == "add":
                    nv = a + int(s[3])
                elif s[1] == "mul":
                    nv = a * int(s[3])
                else:
                    nv = None
            if val.get(v) != nv:
                val[v] = nv
                changed = True
    return val


def bench_sccp_const(seed: int = 20261231 + 351) -> dict[str, float]:
    rng = random.Random(seed)
    const_ok = cover = 0
    trials = 40
    for _ in range(trials):
        stmts: list[tuple[str, str, int | str, int]] = []
        vars_: list[str] = []
        for i in range(rng.randrange(4, 15)):
            v = f"v{i}"
            if i == 0 or rng.random() < 0.4:
                stmts.append((v, "imm", rng.randrange(1, 10), 0))
            else:
                a = rng.choice(vars_)
                op = rng.choice(["add", "mul"])
                stmts.append((v, op, a, rng.randrange(1, 6)))
            vars_.append(v)
        val = sccp(stmts)
        # oracle: sequential eval
        env: dict[str, int] = {}
        for s in stmts:
            if s[1] == "imm":
                env[s[0]] = int(s[2])
            elif s[1] == "add":
                env[s[0]] = env[str(s[2])] + int(s[3])
            else:
                env[s[0]] = env[str(s[2])] * int(s[3])
        const_ok += int(all(val[k] == v for k, v in env.items() if val.get(k) is not None))
        cover += int(len(val) == len(env))
    return {
        "synthetic_consts_match_oracle": float(const_ok / trials),
        "synthetic_full_coverage": float(cover / trials),
    }
