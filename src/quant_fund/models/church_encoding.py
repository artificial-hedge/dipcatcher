"""Church encodings: booleans, pairs, numerals evaluated on concrete terms (SYNTHETIC)."""

from __future__ import annotations

from collections.abc import Callable

# Church booleans as python functions (faithful encoding)
TRUE = lambda t, f: t  # noqa: E731
FALSE = lambda t, f: f  # noqa: E731
AND = lambda a, b: a(b, FALSE)  # noqa: E731
OR = lambda a, b: a(TRUE, b)  # noqa: E731
NOT = lambda a: a(FALSE, TRUE)  # noqa: E731
IF = lambda c, t, e: c(t, e)  # noqa: E731


def ch_num(n: int) -> Callable:
    def num(f, x):
        for _ in range(n):
            x = f(x)
        return x

    return num


def ch_add(m, n):
    return lambda f, x: m(f, n(f, x))


def ch_mul(m, n):
    return lambda f, x: m(lambda y: n(f, y), x)


def ch_iszero(n):
    return n(lambda _x: FALSE, TRUE)


def to_int(n) -> int:
    return int(n(lambda x: x + 1, 0))


def _bench_church_encoding(seed: int = 0) -> float:
    checks = []
    checks.append(AND(TRUE, TRUE)(1, 0) == 1 and AND(TRUE, FALSE)(1, 0) == 0)
    checks.append(OR(FALSE, TRUE)(1, 0) == 1)
    checks.append(NOT(TRUE)(1, 0) == 0)
    checks.append(IF(FALSE, "t", "e") == "e")
    checks.append(to_int(ch_num(4)) == 4)
    checks.append(to_int(ch_add(ch_num(3), ch_num(4))) == 7)
    checks.append(to_int(ch_mul(ch_num(3), ch_num(4))) == 12)
    checks.append(ch_iszero(ch_num(0))(1, 0) == 1 and ch_iszero(ch_num(2))(1, 0) == 0)
    return float(sum(checks) / len(checks))


def bench_church_encoding(seed: int = 0) -> dict[str, float]:
    return {"synthetic_church_encoding": _bench_church_encoding(seed)}
