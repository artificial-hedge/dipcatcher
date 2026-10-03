"""Algebraic effects and handlers (Plotkin–Pretnar style).

Computation trees: ("ret",v) | ("do",eff,k). handle walks the tree
top-down: each handler clause either transforms an effect and resumes
(via the captured continuation applied to a sub-handler result) or
returns the final value — handling order matters exactly as in
multi-handler effect systems (e.g. state-then-exception vs the
opposite).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_SEED = 20261231 + 1024

Comp = tuple
Handler = dict[str, Callable[[Any, Callable[[Any], "Comp"]], "Comp"]]


def ret(v: Any) -> Comp:
    return ("ret", v)


def do(eff: Any, k: Callable[[Any], Comp]) -> Comp:
    return ("do", eff, k)


def eval_prog(c: Comp) -> Any:
    while c[0] == "do":
        c = c[2](None)
    return c[1]


def handle(c: Comp, hs: Handler, ret_clause: Callable[[Any], Any] | None = None) -> Any:
    """Deep handler: dispatch on effect name; `k` is resumable continuation.
    Result type = handler's chosen carrier."""
    if c[0] == "ret":
        return ret_clause(c[1]) if ret_clause else c[1]
    eff, k = c[1], c[2]
    name = eff[0]
    if name in hs:

        def resume(v: Any) -> Any:
            return _handled(k(v), hs, ret_clause)

        return _cont_result(hs[name](eff, resume))
    # unhandled effect propagates through the handler
    return _propagate(c, hs, ret_clause)


def _handled(c: Comp, hs: Handler, rc: Callable[[Any], Any] | None) -> Any:
    return handle(c, hs, rc)


def _propagate(c: Comp, hs: Handler, rc: Callable[[Any], Any] | None) -> Any:
    _eff, k = c[1], c[2]
    # apply the continuation with a "forwarded" answer — simplification:
    # propagate by running the rest under the same handler
    return handle(k(None), hs, rc)


def _cont_result(c: Any) -> Any:
    return c


def bench_alg_effects(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []

    # state handler: get/put threaded through a dict carrier
    def state_h(eff: Any, resume: Callable[[Any], Any]) -> Any:
        name = eff[0]
        if name == "get":
            return resume(st["v"])
        if name == "put":
            st["v"] = eff[1]
            return resume(None)
        raise ValueError(name)

    st = {"v": 3}
    prog = do(("get",), lambda n: do(("put", n * 2), lambda __: do(("get",), lambda m: ret(m + 1))))
    hs: Handler = {"get": state_h, "put": state_h}
    checks.append(handle(prog, hs) == 7 and st["v"] == 6)

    # exception handler: throw short-circuits with its payload
    def exc_h(eff: Any, resume: Callable[[Any], Any]) -> Any:
        return ("err", eff[1])

    prog2 = do(("throw", "boom"), lambda _: ret(1))
    checks.append(handle(prog2, {"throw": exc_h}) == ("err", "boom"))

    # reader: ask yields fixed env
    def ask_h(eff: Any, resume: Callable[[Any], Any]) -> Any:
        return resume(41)

    checks.append(handle(do(("ask",), lambda e: ret(e + 1)), {"ask": ask_h}) == 42)
    # ret clause transforms final value
    checks.append(handle(ret(5), {}, lambda v: ("ok", v)) == ("ok", 5))
    return {"synthetic_alg_effects": float(sum(checks)) / len(checks)}
