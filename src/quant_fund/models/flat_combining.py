"""Flat combining: combiner executes published ops — sequential-equivalence (SYNTHETIC)
oracle."""

import numpy as np

_SEED = 20261231 + 584


def _combine(ops: list[tuple[str, int]]) -> list[int]:
    state: list[int] = []
    results: list[int] = []
    for op, v in ops:
        if op == "push":
            state.append(v)
            results.append(-1)
        else:
            results.append(state.pop() if state else -1)
    return results


def bench_flat_combining(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(30):
        ops = [("push", int(v)) if rng.rand() < 0.5 else ("pop", 0) for v in rng.randint(0, 99, 80)]
        combined = _combine(ops)
        # sequential oracle: stack semantics — pop returns last pushed
        state: list[int] = []
        oracle = []
        for op, v in ops:
            if op == "push":
                state.append(v)
                oracle.append(-1)
            else:
                oracle.append(state.pop() if state else -1)
        if combined == oracle:
            ok += 1
    return {"synthetic_combining_equiv": ok / 30}
