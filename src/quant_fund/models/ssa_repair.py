"""Phi elimination via parallel copies (out-of-SSA repair).

A phi at a block header merges values per predecessor. To lower it,
emit a parallel copy at each predecessor edge; cycles among the copies
need a scratch register (swap cycle break).
"""

from __future__ import annotations

_SEED = 20261231 + 900

Copy = tuple[str, str]  # (dst, src)


def schedule_copies(copies: list[Copy]) -> list[list[str]]:
    """Order parallel copies; break cycles with a scratch temp.

    Returns instruction list of ("mov", dst, src) 3-tuples.
    """
    pending = {d: s for d, s in dict(copies).items() if d != s}
    out: list[list[str]] = []
    while pending:
        srcs = set(pending.values())
        free = [d for d in pending if d not in srcs]
        if free:
            d = free[0]
            out.append(["mov", d, pending.pop(d)])
            continue
        # all pending form cycles: save a dst value into scratch first
        d0 = next(iter(pending))
        out.append(["mov", "_scratch", d0])
        for d in list(pending):
            if pending[d] == d0:
                pending[d] = "_scratch"
    return out


def exec_copies(init: dict[str, int], seq: list[list[str]]) -> dict[str, int]:
    env = dict(init)
    for _, d, s in seq:
        env[d] = env.get(s, 0) if s != "_scratch" else env.get("_scratch", 0)
    return env


def parallel_semantics(init: dict[str, int], copies: list[Copy]) -> dict[str, int]:
    """True parallel semantics: read all sources first."""
    env = dict(init)
    vals = {d: env.get(s, 0) for d, s in copies}
    env.update(vals)
    return env


def bench_ssa_repair(seed: int = _SEED) -> dict[str, float]:
    score = 0.0
    # simple acyclic chain: a<-b, b<-c
    seq = schedule_copies([("a", "b"), ("b", "c")])
    env = exec_copies({"a": 0, "b": 1, "c": 2}, seq)
    score += 1.0 if env["a"] == 1 and env["b"] == 2 else 0.0
    # swap cycle: x<-y, y<-x needs scratch
    seq2 = schedule_copies([("x", "y"), ("y", "x")])
    env2 = exec_copies({"x": 5, "y": 7, "_scratch": 0}, seq2)
    score += 1.0 if env2["x"] == 7 and env2["y"] == 5 else 0.0
    # 3-cycle: a<-b, b<-c, c<-a
    seq3 = schedule_copies([("a", "b"), ("b", "c"), ("c", "a")])
    env3 = exec_copies({"a": 1, "b": 2, "c": 3, "_scratch": 0}, seq3)
    ref = parallel_semantics({"a": 1, "b": 2, "c": 3}, [("a", "b"), ("b", "c"), ("c", "a")])
    score += 1.0 if all(env3[k] == ref[k] for k in "abc") else 0.0
    # scratch actually used for cycles
    score += 1.0 if any(m[2] == "_scratch" or m[1] == "_scratch" for m in seq2) else 0.0
    return {"synthetic_ssa_repair": score / 4.0}
