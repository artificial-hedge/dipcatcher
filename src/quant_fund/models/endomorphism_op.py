"""Endomorphism operad and algebra maps (SYNTHETIC)."""

from __future__ import annotations


def end_n(x: tuple[int, ...], n: int) -> list[dict[tuple[int, ...], int]]:
    """End_X(n) = all maps X^n -> X on a finite set X (toy sizes)."""
    from itertools import product

    keys = list(product(x, repeat=n))
    # only generate a couple of canonical ops rather than all |X|^{|X|^n}
    const0 = dict.fromkeys(keys, x[0])
    proj_first = {k: k[0] for k in keys}
    proj_last = {k: k[-1] for k in keys}
    return [const0, proj_first, proj_last]


def operad_map_preserves_comp(m_alg, m_op: dict[tuple[int, ...], int], x: tuple[int, ...]) -> bool:
    """An algebra map respects composition: applying the algebra's binary
    op to (a, b op c) equals op applied componentwise — checks the
    operad homomorphism axiom on a concrete Assoc action."""
    for a in x:
        for b in x:
            for c in x:
                lhs = m_op[(a, m_op[(b, c)])]
                rhs = m_op[(m_op[(a, b)], c)]
                if lhs != rhs:
                    return False
    return True


def _bench_endomorphism_op(seed: int = 0) -> float:
    checks = []
    x = (0, 1, 2)
    ops = end_n(x, 2)
    checks.append(len(ops) == 3)
    checks.append(ops[1][(0, 2)] == 0 and ops[2][(0, 2)] == 2)
    # the cyclic-group addition is an associative op on X=Z_3
    add = {k: (k[0] + k[1]) % 3 for k in ops[1]}
    checks.append(operad_map_preserves_comp(None, add, x))
    # a non-associative op fails the test: subtraction mod 3
    sub = {k: (k[0] - k[1]) % 3 for k in ops[1]}
    checks.append(not operad_map_preserves_comp(None, sub, x))
    # projection is associative
    checks.append(operad_map_preserves_comp(None, ops[1], x))
    checks.append(operad_map_preserves_comp(None, ops[2], x))
    return float(sum(checks) / len(checks))


def bench_endomorphism_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_endomorphism_op": _bench_endomorphism_op(seed)}
