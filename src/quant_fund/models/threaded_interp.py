"""SYNTHETIC threaded-code interpreter (direct-dispatch function table).

Same bytecode, two interpreters: switch-dispatch vs handler-table
threaded dispatch — verified identical states and results.
"""

from __future__ import annotations

import random

_OPS = ["push", "add", "dup", "swap"]


def _gen_prog(rng: random.Random, n: int) -> list[tuple[str, ...] | tuple[str, int]]:
    out: list[tuple[str, ...] | tuple[str, int]] = []
    depth = 0
    for _ in range(n):
        if depth < 2 or rng.random() < 0.5:
            out.append(("push", rng.randrange(9)))
            depth += 1
        else:
            op = rng.choice(["add", "dup", "swap"])
            out.append((op,))
            depth += 1 if op == "dup" else (-1 if op == "add" else 0)
    while depth < 2:
        out.append(("push", 0))
        depth += 1
    out.append(("add",))
    return out


def run_switch(code: list[tuple]) -> list[int]:
    st: list[int] = []
    for ins in code:
        op = ins[0]
        if op == "push":
            st.append(ins[1])
        elif op == "add":
            b, a = st.pop(), st.pop()
            st.append(a + b)
        elif op == "dup":
            st.append(st[-1])
        elif op == "swap":
            st[-1], st[-2] = st[-2], st[-1]
        else:
            raise ValueError(op)
    return st


def _h_push(st: list[int], ins: tuple) -> None:
    st.append(ins[1])


def _h_add(st: list[int], ins: tuple) -> None:
    del ins
    b, a = st.pop(), st.pop()
    st.append(a + b)


def _h_dup(st: list[int], ins: tuple) -> None:
    del ins
    st.append(st[-1])


def _h_swap(st: list[int], ins: tuple) -> None:
    del ins
    st[-1], st[-2] = st[-2], st[-1]


HANDLERS = {"push": _h_push, "add": _h_add, "dup": _h_dup, "swap": _h_swap}


def run_threaded(code: list[tuple]) -> list[int]:
    st: list[int] = []
    for ins in code:
        HANDLERS[ins[0]](st, ins)
    return st


def bench_threaded_interp(seed: int = 20261231 + 481) -> dict[str, float]:
    rng = random.Random(seed)
    match = cover = stable = 0
    trials = 60
    for _ in range(trials):
        prog = _gen_prog(rng, rng.randrange(3, 12))
        match += int(run_switch(prog) == run_threaded(prog))
        cover += int(set(HANDLERS) == set(_OPS))
        stable += int(run_threaded(prog) == run_threaded(prog))
    return {
        "synthetic_dispatch_equivalent": float(match / trials),
        "synthetic_handler_coverage": float(cover / trials),
        "synthetic_deterministic": float(stable / trials),
    }
