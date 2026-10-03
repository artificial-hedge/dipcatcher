"""Loop-invariant code motion — SYNTHETIC.

Loop: (header_consts, body_ops, trips). Body ops use loop vars and
invariant vars; invariants' defs get hoisted. Verified: identical
loop result; hoisted op count > 0 when invariants exist.
"""

from __future__ import annotations

import random


def run_loop(
    pre: list[tuple[str, str, int | str, int | str]],  # (out, op, a, b)
    body: list[tuple[str, str, int | str, int | str]],
    trips: int,
    env: dict[str, int],
    hoist: bool,
) -> dict[str, int]:
    def ev(stmts: list[tuple[str, str, int | str, int | str]], e: dict[str, int]) -> None:
        for o, op, a, b in stmts:
            av = e[a] if isinstance(a, str) else a
            bv = e[b] if isinstance(b, str) else b
            e[o] = av + bv if op == "add" else av * bv

    ev(pre, env)
    if hoist:
        # pull body ops whose operands don't depend on 'i' or body defs
        body_names = {s[0] for s in body} | {"i"}
        invar = [
            s
            for s in body
            if (not isinstance(s[2], str) or s[2] not in body_names)
            and (not isinstance(s[3], str) or s[3] not in body_names)
        ]
        rest = [s for s in body if s not in invar]
        ev(invar, env)  # hoisted once before loop (inputs loop-stable)
        for _ in range(trips):
            env["i"] = env.get("i", 0)
            ev(rest, env)
            env["i"] += 1
        return env
    for _ in range(trips):
        env["i"] = env.get("i", 0)
        ev(body, env)
        env["i"] += 1
    return env


def bench_licm_hoist(seed: int = 20261231 + 355) -> dict[str, float]:
    rng = random.Random(seed)
    same = hoisted = 0
    trials = 40
    for _ in range(trials):
        trips = rng.randrange(2, 8)
        env = {f"x{i}": rng.randrange(1, 10) for i in range(3)}
        body: list[tuple[str, str, int | str, int | str]] = [
            ("t1", "mul", "x0", "x1"),  # invariant
            ("acc", "add", "acc", "t1"),  # uses t1 + acc (loop var)
            ("i2", "add", "i", 1),  # uses i
        ]
        e1 = dict(env)
        e1["acc"] = 0
        r1 = run_loop([], body, trips, dict(e1), hoist=False)
        e2 = dict(env)
        e2["acc"] = 0
        r2 = run_loop([], body, trips, dict(e2), hoist=True)
        same += int(r1["acc"] == r2["acc"])
        hoisted += int(True)  # metric: invar detection found t1
        body_names = {s[0] for s in body} | {"i"}
        invar = [
            s
            for s in body
            if (not isinstance(s[2], str) or s[2] not in body_names)
            and (not isinstance(s[3], str) or s[3] not in body_names)
        ]
        hoisted -= 1
        hoisted += int(any(s[0] == "t1" for s in invar))
    return {
        "synthetic_result_identical": float(same / trials),
        "synthetic_invariant_detected": float(hoisted / trials),
    }
