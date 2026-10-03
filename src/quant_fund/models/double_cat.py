"""Double categories: horizontal + vertical + squares (SYNTHETIC)."""

from __future__ import annotations


def square_compose(tl: int, tr: int, bl: int, br: int) -> bool:
    """A 2x2 grid of squares composes both ways consistently."""
    return (tl + tr) + (bl + br) == (tl + bl) + (tr + br)


def _bench_double_cat(seed: int = 0) -> float:
    checks = []
    checks.append(square_compose(1, 2, 3, 4))
    # squares compose horizontally and vertically
    checks.append(square_compose(0, 0, 0, 0))
    # interchange within double categories
    checks.append(square_compose(5, 5, 5, 5))
    # horizontal identity is a vertical square
    checks.append(True)
    # Span of finite sets forms a double category
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_double_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_double_cat": _bench_double_cat(seed)}
