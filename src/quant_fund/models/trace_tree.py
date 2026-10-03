"""Trace-tree JIT simulation: record loop iterations, guard side-exits.

A trace is a linearized recording of a hot loop body. Guards check
assumptions (types/branches); a guard failure takes a side exit back to
the interpreter. Side traces compile off failing guards.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 899


def simulate(trace_len: int, iterations: list[str]) -> dict[str, int]:
    """Simulate a one-trace JIT over a loop with two value branches.

    `iterations`: sequence of "a"|"b" path choices per loop iteration.
    The recorded trace follows the first iteration's path; every
    subsequent iteration matching the recorded path stays on-trace,
    mismatches take a side exit (and after `trace_len` hot side-exits a
    side trace would form).
    Returns counts: on_trace, side_exit, side_traces.
    """
    if not iterations:
        return {"on_trace": 0, "side_exit": 0, "side_traces": 0}
    recorded = iterations[0]
    on_trace = 1
    side = 0
    side_traces = 0
    pending: dict[str, int] = {}
    compiled: set[str] = {recorded}
    for it in iterations[1:]:
        if it in compiled:
            on_trace += 1
            continue
        side += 1
        pending[it] = pending.get(it, 0) + 1
        if pending[it] >= trace_len:
            compiled.add(it)
            side_traces += 1
    return {"on_trace": on_trace, "side_exit": side, "side_traces": side_traces}


def bench_trace_tree(seed: int = _SEED) -> dict[str, float]:
    # 90% 'a' path, 10% 'b' path; side trace forms after 5 b's.
    rng = np.random.default_rng(seed)
    n = 1000
    its = list(np.where(rng.random(n) < 0.9, "a", "b"))
    its[0] = "a"  # ensure 'a' recorded
    res = simulate(5, its)
    score = 0.0
    score += 1.0 if res["side_traces"] == 1 else 0.0
    score += 1.0 if res["side_exit"] == 5 else 0.0  # exits until 'b' side trace compiles
    # accounting identity: on_trace + side_exit == n
    score += 1.0 if res["on_trace"] + res["side_exit"] == n else 0.0
    # pure-a run has zero side exits
    res2 = simulate(5, ["a"] * 100)
    score += 1.0 if res2 == {"on_trace": 100, "side_exit": 0, "side_traces": 0} else 0.0
    return {"synthetic_trace_tree": score / 4.0}
