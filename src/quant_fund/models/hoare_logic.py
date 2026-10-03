"""Hoare-logic triple verifier on a tiny while-language (synthetic).

Programs are tuples of statements; the verifier computes the weakest
liberal precondition through assignments and (annotated) while-loops
via predicate transformers over integer arithmetic, then checks
``pre => wlp(program, post)`` by exhaustive bounded evaluation.
Correctness claims are cross-checked by concrete execution.
"""

from __future__ import annotations

from typing import Any


def _wlp(stmt: Any, post, env_bounds) -> Any:
    """Weakest liberal precondition; ``post`` is a predicate fn(env)->bool."""
    kind = stmt[0]
    if kind == "assign":
        _, var, expr_fn = stmt
        return lambda env: post({**env, var: expr_fn(env)})
    if kind == "seq":
        mid = post
        for s in reversed(stmt[1]):
            mid = _wlp(s, mid, env_bounds)
        return mid
    if kind == "while":
        # stmt = ("while", guard_fn, body, invariant_pred)
        _, g_fn, body, inv = stmt
        # wlp of loop: invariant must hold, and (inv & !g) => post
        body_wlp = _wlp(("seq", body), inv, env_bounds)

        def check(env) -> bool:
            if not inv(env):
                return False
            return bool(g_fn(env) or post(env))

        def full(env) -> bool:
            if not check(env):
                return False
            # verify body preserves invariant on bounded env domain
            for x in range(env_bounds["x"][0], env_bounds["x"][1] + 1):
                for y in range(env_bounds["y"][0], env_bounds["y"][1] + 1):
                    e = {"x": x, "y": y}
                    if inv(e) and g_fn(e) and not body_wlp(e):
                        return False
            return True

        return full
    raise ValueError(kind)


def _run(prog, env: dict[str, int], fuel: int = 500) -> dict[str, int]:
    for stmt in prog:
        if stmt[0] == "assign":
            env = dict(env)
            env[stmt[1]] = int(stmt[2](env))
        elif stmt[0] == "while":
            _, g, body, _inv = stmt
            f = fuel
            while g(env) and f > 0:
                env = _run(list(body), env, fuel)
                f -= 1
    return env


def bench_hoare_logic(seed: int = 20261231 + 225) -> dict[str, float]:
    # Program: {x=0,y=N} y:=0; while x<N: x:=x+1; y:=y+2  ==> post y=2N
    N = 6
    prog = (
        "seq",
        [
            ("assign", "y", lambda e: 0),
            (
                "while",
                lambda e: e["x"] < N,
                [
                    ("assign", "x", lambda e: e["x"] + 1),
                    ("assign", "y", lambda e: e["y"] + 2),
                ],
                lambda e: e["y"] == 2 * e["x"] and e["x"] <= N,
            ),
        ],
    )
    bounds = {"x": (0, N + 2), "y": (0, 2 * N + 4)}
    post = lambda e: e["y"] == 2 * N and e["x"] == N  # noqa: E731

    wlp = _wlp(prog, post, bounds)
    valid = all(wlp({"x": 0, "y": y}) for y in range(bounds["y"][0], bounds["y"][1] + 1))
    end = _run([("assign", "y", lambda e: 0)] + [prog[1][1]], {"x": 0, "y": 0})
    concrete_ok = bool(post(end))

    # negative: weaker (false) invariant must be rejected
    prog_bad = (
        "seq",
        [
            ("assign", "y", lambda e: 0),
            (
                "while",
                lambda e: e["x"] < N,
                [
                    ("assign", "x", lambda e: e["x"] + 1),
                    ("assign", "y", lambda e: e["y"] + 2),
                ],
                lambda e: e["y"] == 3 * e["x"],  # wrong invariant
            ),
        ],
    )
    wlp_bad = _wlp(prog_bad, post, bounds)
    bad_rejected = not all(
        wlp_bad({"x": 0, "y": y}) for y in range(bounds["y"][0], bounds["y"][1] + 1)
    )
    return {
        "synthetic_wlp_valid": float(valid),
        "synthetic_concrete_ok": float(concrete_ok),
        "synthetic_agree": float(valid == concrete_ok),
        "synthetic_bad_rejected": float(bad_rejected),
    }
