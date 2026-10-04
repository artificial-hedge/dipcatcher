"""Free monad over a functor + stack-safe interpreters.

Terms: ("pure",v) | ("op",f,k) | ("bind",m,k). run folds the structure
via an explicit continuation stack (trampolined) so arbitrarily deep
bind chains don't hit Python's recursion limit. Interpreters handle
each functor op and feed a result into the op's continuation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_SEED = 20261231 + 1023


def pure(v: Any) -> tuple:
    return ("pure", v)


def op(f: Any, k: Callable[[Any], tuple] | None = None) -> tuple:
    return ("op", f, k)


def bind(m: tuple, k: Callable[[Any], tuple]) -> tuple:
    return ("bind", m, k)


def run(m: tuple, interp: Callable[[Any], Any]) -> Any:
    """Trampolined fold: interp(f) returns the value fed to the op's
    continuation."""
    stack: list[Callable[[Any], tuple]] = []
    while True:
        while m[0] == "bind":
            stack.append(m[2])
            m = m[1]
        if m[0] == "pure":
            v = m[1]
        elif m[0] == "op":
            v = interp(m[1])
            if m[2] is not None:
                m = m[2](v)
                continue
        else:
            raise ValueError(m)
        if not stack:
            return v
        m = stack.pop()(v)


def bench_free_monad(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # writer: tell "a" >> tell "b" >> return 7
    log: list[str] = []

    def w_interp(f: Any) -> Any:
        if f[0] == "tell":
            log.append(f[1])
        return f[1]

    prog = bind(
        op(("tell", "a")),
        lambda _: bind(op(("tell", "b")), lambda __: pure(7)),
    )
    v = run(prog, w_interp)
    checks.append(v == 7 and log == ["a", "b"])

    # deep chain: 5000 binds — stack-safe (recursion limit unaffected)
    def _step_maker(i: int) -> Callable[[Any], tuple]:
        def _step(v: Any) -> tuple:
            return pure(v + 1 if i % 2 else v)

        return _step

    m = pure(0)
    for i in range(5000):
        m = bind(m, _step_maker(i))
    checks.append(run(m, w_interp) == 2500)
    # state effect: get then put
    st = {"n": 0}

    def s_interp(f: Any) -> Any:
        if f[0] == "get":
            return st["n"]
        st["n"] = f[1]
        return f[1]

    sprog = bind(
        op(("get",)),
        lambda n: bind(op(("put", n + 5)), lambda __: pure(st["n"])),
    )
    v3 = run(sprog, s_interp)
    checks.append(v3 == 5 and st["n"] == 5)
    # op with continuation feeds interp result through
    v4 = run(op(("tell", "z"), lambda e: pure(e)), w_interp)
    checks.append(v4 == "z" and log[-1] == "z")
    return {"synthetic_free_monad": float(sum(checks)) / len(checks)}
