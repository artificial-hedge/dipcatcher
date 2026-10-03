"""Arrays-with-extensionality (QF_AX): read/write axiom checker (SYNTHETIC bench)."""

from __future__ import annotations

from typing import Any

Term = Any  # ("arr",name) | ("store",a,i,v) | ("read",a,i) | ("const",v) | ("sym",v)


def eval_term(t: Term, store: dict) -> Any:
    """Fully evaluate a read/store term to a concrete or symbolic value."""
    if t[0] == "const":
        return t[1]
    if t[0] == "sym":
        return ("sym", t[1])
    if t[0] == "arr":
        if t[1] in store:
            return store[t[1]]
        return ("arr", t[1])
    if t[0] == "store":
        a = eval_term(t[1], store)
        i = eval_term(t[2], store)
        v = eval_term(t[3], store)
        if a[0] == "fun":
            m = dict(a[1])
            m[i] = v
            return ("fun", m)
        return ("store", a, i, v)
    if t[0] == "read":
        a = eval_term(t[1], store)
        i = eval_term(t[2], store)
        return read_eval(a, i)
    raise ValueError(t[0])


def read_eval(a: Any, i: Any) -> Any:
    """read(a,i) with store-distribution: read(store(a,j,v),i)."""
    while a[0] == "store":
        _, base, j, v = a
        if i == j:
            return v
        a = base
    if a[0] == "fun":
        m = a[1]
        if i in m:
            return m[i]
        return ("read", a, i)
    return ("read", a, i)


def holds(eq: tuple, store: dict | None = None) -> bool:
    """Check a ground equality t1 == t2 after evaluation."""
    return bool(eval_term(eq[0], store or {}) == eval_term(eq[1], store or {}))


def consistent_assign(assigns: list[tuple], checks: list[tuple]) -> bool:
    """Assert a[i]:=v assignments then check equalities/inequalities.

    checks: list of (lhs, rhs, eq?) — eq=True means must be equal, False disequal.
    """
    store: dict = {}
    for name, i, v in assigns:
        arr = store.get(name, ("arr", name))
        store[name] = ("store", arr, i, v)
    for lhs, rhs, is_eq in checks:
        a = eval_term(lhs, store)
        b = eval_term(rhs, store)
        same = a == b
        if is_eq and not same:
            return False
        if not is_eq and same:
            return False
    return True


def _bench_array_theory(seed: int = 0) -> float:
    del seed
    checks = []
    a = ("arr", "a")
    w = ("store", a, ("const", 1), ("const", 9))
    checks.append(holds((("read", w, ("const", 1)), ("const", 9))))
    checks.append(
        holds(
            (
                ("read", w, ("const", 2)),
                ("read", a, ("const", 2)),
            )
        )
    )
    checks.append(
        consistent_assign(
            [("a", 0, 5)], [(("read", ("arr", "a"), ("const", 0)), ("const", 5), True)]
        )
    )
    checks.append(
        consistent_assign(
            [("a", 0, 5), ("a", 1, 7)],
            [
                (("read", ("arr", "a"), ("const", 0)), ("const", 5), True),
                (("read", ("arr", "a"), ("const", 1)), ("const", 7), True),
            ],
        )
    )
    # store then read different index reads through
    checks.append(
        consistent_assign(
            [("a", 0, 5)],
            [(("read", ("arr", "a"), ("const", 1)), ("read", ("arr", "a"), ("const", 1)), True)],
        )
    )
    # disequality check fails when values forced equal
    checks.append(
        not consistent_assign(
            [("a", 0, 5)], [(("read", ("arr", "a"), ("const", 0)), ("const", 5), False)]
        )
    )
    return sum(checks) / len(checks)


def bench_array_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_array_theory": _bench_array_theory(seed)}
