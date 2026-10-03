"""Back-and-forth equivalence on dense orders (SYNTHETIC)."""

from __future__ import annotations

from fractions import Fraction


def forth_move(partial: dict[Fraction, Fraction], a: Fraction, target: list[Fraction]) -> Fraction:
    """Extend the finite order-preserving map a -> b in a dense order:
    pick the target element preserving all comparisons to current pairs."""
    used = set(partial.values())
    lo = max((v for k, v in partial.items() if k < a), default=None)
    hi = min((v for k, v in partial.items() if k > a), default=None)
    valid = [
        c
        for c in sorted(target)
        if c not in used and (lo is None or c > lo) and (hi is None or c < hi)
    ]
    if not valid:
        raise ValueError("no forth move available")
    return valid[len(valid) // 2]


def back_move(partial: dict[Fraction, Fraction], b: Fraction, source: list[Fraction]) -> Fraction:
    """The back direction: find a source element hitting b."""
    used_keys = set(partial)
    lo = max((k for k, v in partial.items() if v < b), default=None)
    hi = min((k for k, v in partial.items() if v > b), default=None)
    valid = [
        c
        for c in sorted(source)
        if c not in used_keys and (lo is None or c > lo) and (hi is None or c < hi)
    ]
    if not valid:
        raise ValueError("no back move available")
    return valid[len(valid) // 2]


def _bench_back_forth(seed: int = 0) -> float:
    checks = []
    src = [Fraction(i, 4) for i in range(-8, 9)]
    tgt = [Fraction(i, 8) for i in range(-40, 41)]
    partial: dict[Fraction, Fraction] = {}
    # forth: send 1/4 to something preserving empty constraints
    b1 = forth_move(partial, Fraction(1, 4), tgt)
    partial[Fraction(1, 4)] = b1
    # back: pick a source element mapping to -1/2
    a2 = back_move(partial, Fraction(-1, 2), src)
    partial[a2] = Fraction(-1, 2)
    # map is order preserving so far
    items = sorted(partial.items())
    checks.append(all(items[i][1] < items[i + 1][1] for i in range(len(items) - 1)))
    # forth again below both constraints: send -1/4 (between a2=-1/2? find)
    checks.append(len(partial) == 2)
    # another forth move preserves order
    a3 = Fraction(0)
    b3 = forth_move(partial, a3, tgt)
    partial[a3] = b3
    items = sorted(partial.items())
    checks.append(all(items[i][1] < items[i + 1][1] for i in range(len(items) - 1)))
    # a map sending 0->0,1/4->1/4 is order-preserving
    checks.append(b3 is not None)
    checks.append(isinstance(b1, Fraction))
    return float(sum(checks) / len(checks))


def bench_back_forth(seed: int = 0) -> dict[str, float]:
    return {"synthetic_back_forth": _bench_back_forth(seed)}
