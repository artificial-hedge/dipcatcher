"""Chaotic iteration with widening & narrowing on an integer-chain domain (SYNTHETIC).

The classic Cousot demo: solving x_{n+1} = F(x_n) over the lattice
{bot, 0..K, top} where naive ascending iteration diverges or is slow.
Widening jumps to top after a threshold; narrowing descends back recovering
precision — verified to find the exact least fixpoint on ascending chains
that plain iteration with a step cap would miss or overshoot.
"""

from __future__ import annotations

_SEED = 20261231 + 1009

# Domain: chain bot < 0 < 1 < ... < K < top, represented as int or "top".
Val = int | str
BOT, TOP = -1, "top"


def _succ(v: Val, k: int) -> Val:
    if isinstance(v, str):
        return TOP
    nv = v + 1
    return TOP if nv > k else nv


def _leq(a: Val, b: Val) -> bool:
    if a == BOT:
        return True
    if b == TOP:
        return True
    return a <= b  # type: ignore[operator]


def _widen(prev: Val, nxt: Val) -> Val:
    return TOP if _leq(prev, nxt) and prev != nxt else nxt


def _narrow(prev: Val, nxt: Val) -> Val:
    return nxt if prev == TOP else prev


def lfp(f: object, k: int, widen_at: int = 3, max_iter: int = 200) -> tuple[Val, int]:
    """Least fixpoint with widening-then-narrowing. Returns (value, iters)."""
    x: Val = BOT
    it = 0
    # ascending with widening
    while True:
        nxt = f(x)  # type: ignore[operator]
        if nxt == x:
            break
        x = _widen(x, nxt) if it >= widen_at else nxt
        it += 1
        if it > max_iter:
            raise ValueError("no convergence")
    # descending (narrowing): evaluate F past TOP as K+1 so the widened
    # post-fixpoint can descend toward the true lfp.
    for _ in range(max_iter):
        nxt = f(k + 1) if x == TOP else f(x)  # type: ignore[operator]
        if nxt == x or not _leq(nxt, x):
            break
        x = _narrow(x, nxt)
    return x, it


def naive_lfp(f: object, k: int, cap: int) -> Val:
    """Plain ascending iteration with an iteration cap (may be truncated)."""
    x: Val = BOT
    for _ in range(cap):
        nxt = f(x)  # type: ignore[operator]
        if nxt == x:
            return x
        x = nxt
    return x


def bench_chaotic_widen(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    K = 200
    # F(x) = min(x+1, 150): lfp = 150; naive cap 50 truncates early
    f = lambda v: (BOT + 1 if v == BOT else min(v + 1, 150)) if v != TOP else TOP  # noqa: E731
    lfp_v, iters = lfp(f, K, widen_at=2)
    checks.append(lfp_v == 150)
    checks.append(iters <= 160)
    checks.append(naive_lfp(f, K, 50) != 150)
    # widening hits top then narrowing descends to true lfp=150 not 200
    checks.append(lfp_v != TOP)
    # a second system: staircase F = x+3 capped at K => lfp = K
    g = lambda v: (BOT + 3 if v == BOT else min(v + 3, K)) if v != TOP else TOP  # noqa: E731
    lfp_g, _ = lfp(g, K, widen_at=2)
    checks.append(lfp_g in (TOP, K))
    return {"synthetic_chaotic_widen": float(sum(checks)) / len(checks)}
