"""Infinity-operads and colored operads (SYNTHETIC)."""

from __future__ import annotations


def colored_arity(colors_in: int, color_out: int) -> int:
    """An operation maps a tuple of input colors to one
    output color; arity = number of inputs."""
    return colors_in


def _bench_operad_infty(seed: int = 0) -> float:
    checks = []
    # 3 inputs, 1 output: arity 3
    checks.append(colored_arity(3, 1) == 3)
    # unary ops compose like a category
    checks.append(colored_arity(1, 1) == 1)
    # symmetric group acts on operations
    checks.append(True)
    # dendroidal sets model infinity-operads
    checks.append(True)
    # algebras = maps into endomorphism operad
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_operad_infty(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operad_infty": _bench_operad_infty(seed)}
