"""Proper tail calls via trampolining — deep mutual recursion without stack growth (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 545


class Bounce:
    def __init__(self, fn, *args) -> None:
        self.fn = fn
        self.args = args


def trampoline(fn, *args):
    b = Bounce(fn, *args)
    steps = 0
    while isinstance(b, Bounce):
        b = b.fn(*b.args)
        steps += 1
        if steps > 3_000_000:
            raise RuntimeError("too many")
    return b


def _even(n: int):
    if n == 0:
        return True
    return Bounce(_odd, n - 1)


def _odd(n: int):
    if n == 0:
        return False
    return Bounce(_even, n - 1)


def _sum_to(n: int, acc: int = 0):
    if n == 0:
        return acc
    return Bounce(_sum_to, n - 1, acc + n)


def bench_trampoline_tc(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    parity_ok = 0
    for _ in range(10):
        n_ = rng.randint(0, 5000)
        parity_ok += int(trampoline(_even, n_) == (n_ % 2 == 0))
    deep = 200_000
    sum_ok = int(trampoline(_sum_to, deep) == deep * (deep + 1) // 2)
    # oracle: closed form
    tri_ok = int(trampoline(_sum_to, 999) == 999 * 1000 // 2)
    return {
        "synthetic_parity_exact": float(parity_ok / 10),
        "synthetic_deep_recursion": float(sum_ok),
        "synthetic_tail_exact": float(tri_ok),
    }
