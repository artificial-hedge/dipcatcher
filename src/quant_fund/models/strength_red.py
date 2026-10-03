"""Strength reduction: multiply by constant -> shifts + adds."""

import numpy as np

_SEED = 20261231 + 724


def reduce_mul(k: int) -> list[tuple[str, int]]:
    """Decompose x*k into shift/add ops (naiive power-of-two + remainder)."""
    ops: list[tuple[str, int]] = []
    rem = abs(k)
    shift = 0
    first = True
    while rem:
        if rem & 1:
            ops.append(("shl" if first else "add_shl", shift))
            first = False
        rem >>= 1
        shift += 1
    if k < 0:
        ops.append(("neg", 0))
    return ops or [("const", 0)]


def eval_ops(ops: list[tuple[str, int]], x: int) -> int:
    acc = 0
    for name, s in ops:
        if name == "shl":
            acc = x << s
        elif name == "add_shl":
            acc += x << s
        elif name == "neg":
            acc = -acc
        else:
            acc = int(s)
    return acc


def bench_strength_red(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 60
    for _ in range(trials):
        k = int(rng.randint(-63, 63))
        x = int(rng.randint(-100, 100))
        ok += float(eval_ops(reduce_mul(k), x) == x * k)
    return {"synthetic_strength_exact": ok / trials}
