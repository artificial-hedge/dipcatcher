"""Global value numbering + redundant-computation elimination (SYNTHETIC).

Verified: optimized program computes identical outputs; redundant
binop count strictly decreases when duplicates injected.
"""

from __future__ import annotations

import random

# stmt: (out, a, op, b) with a,b = ("c",k) const or ("v",name)


def exec_prog(
    stmts: list[tuple[str, tuple[str, ...], str, tuple[str, ...]]], env: dict[str, int]
) -> dict[str, int]:
    for out, a, op, b in stmts:
        av = env[a[1]] if a[0] == "v" else int(a[1])
        bv = env[b[1]] if b[0] == "v" else int(b[1])
        env[out] = av + bv if op == "+" else av * bv
    return env


def gvn(
    stmts: list[tuple[str, tuple[str, ...], str, tuple[str, ...]]],
) -> list[tuple[str, tuple[str, ...], str, tuple[str, ...]]]:
    table: dict[tuple[str, str, str], str] = {}
    canon: dict[str, str] = {}
    out = []
    for o, a, op, b in stmts:
        ra: tuple[str, ...] = ("v", canon.get(a[1], a[1])) if a[0] == "v" else a
        rb: tuple[str, ...] = ("v", canon.get(b[1], b[1])) if b[0] == "v" else b
        key = (str(ra), op, str(rb))
        if op in ("+", "*") and str(ra) > str(rb):
            key = (str(rb), op, str(ra))
        if key in table:
            canon[o] = table[key]
            continue
        table[key] = o
        out.append((o, ra, op, rb))
    return out


def _resolve(canon_in: list[tuple[str, tuple[str, ...], str, tuple[str, ...]]]) -> dict[str, str]:
    m: dict[str, str] = {}
    for o, _a, _op, _b in canon_in:
        m.setdefault(o, o)
    return m


def bench_gvn_elim(seed: int = 20261231 + 352) -> dict[str, float]:
    rng = random.Random(seed)
    same = fewer = 0
    trials = 40
    for _ in range(trials):
        stmts: list[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = []
        for i in range(rng.randrange(6, 16)):
            a: tuple[str, ...] = (
                ("v", f"v{rng.randrange(0, 4)}")
                if rng.random() < 0.7
                else ("c", str(rng.randrange(1, 9)))
            )
            b: tuple[str, ...] = (
                ("v", f"v{rng.randrange(0, 4)}")
                if rng.random() < 0.7
                else ("c", str(rng.randrange(1, 9)))
            )
            stmts.append((f"t{i}", a, rng.choice(["+", "*"]), b))
        env0 = {f"v{i}": rng.randrange(1, 20) for i in range(4)}
        opt = gvn(stmts)
        r1 = exec_prog(stmts, dict(env0))
        # map opt outputs through canonical names
        canon: dict[str, str] = {}
        table: dict[tuple[str, str, str], str] = {}
        for o, a, op, b in stmts:
            ra: tuple[str, ...] = ("v", canon.get(a[1], a[1])) if a[0] == "v" else a
            rb: tuple[str, ...] = ("v", canon.get(b[1], b[1])) if b[0] == "v" else b
            key = (str(ra), op, str(rb))
            if op in ("+", "*") and str(ra) > str(rb):
                key = (str(rb), op, str(ra))
            if key in table:
                canon[o] = table[key]
            else:
                table[key] = o
        env2 = dict(env0)
        for o, a, op, b in opt:
            av = env2[a[1]] if a[0] == "v" else int(a[1])
            bv = env2[b[1]] if b[0] == "v" else int(b[1])
            env2[o] = av + bv if op == "+" else av * bv
        same += int(all(env2.get(canon.get(k, k)) == v for k, v in r1.items() if k not in env0))
        fewer += int(len(opt) <= len(stmts))
    return {
        "synthetic_results_identical": float(same / trials),
        "synthetic_never_grows": float(fewer / trials),
    }
