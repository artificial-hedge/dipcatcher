"""Context-sensitive Andersen points-to with k-call-string contexts (SYNTHETIC).

Program: functions with alloc/copy/return statements and call sites.
Contexts are bounded call strings (length k); points-to facts are keyed
(context, var). Compares context-sensitive vs monovariant results to show
the merge the former avoids.
"""

from __future__ import annotations

_SEED = 20261231 + 1014

Ctx = tuple[str, ...]
Facts = dict[tuple[Ctx, str], set[str]]


class CtxPTA:
    def __init__(self, k: int = 1) -> None:
        self.k = k
        # funcs: name -> {"allocs":[(var,obj)], "copies":[(dst,src)],
        #                 "calls":[(site, callee, argvars, retvar)], "ret": var|None,
        #                 "formals":[...]}
        self.funcs: dict[str, dict] = {}
        self.pts: Facts = {}

    def add_fn(self, name: str, **spec) -> None:  # type: ignore[no-untyped-def]
        self.funcs[name] = {
            "allocs": spec.get("allocs", []),
            "copies": spec.get("copies", []),
            "calls": spec.get("calls", []),
            "ret": spec.get("ret"),
            "formals": spec.get("formals", []),
        }

    def _ctx(self, caller: Ctx, site: str) -> Ctx:
        if self.k == 0:
            return ()
        return (caller + (site,))[-self.k :]

    def solve(self, entry: str) -> None:
        wl: list[tuple[Ctx, str]] = [((), entry)]
        while wl:
            ctx, fn = wl.pop()
            f = self.funcs[fn]
            for v, o in f["allocs"]:
                obj = f"&{o}"
                if obj not in self.pts.get((ctx, v), set()):
                    self.pts.setdefault((ctx, v), set()).add(obj)
                    wl.append((ctx, fn))
            for d, s in f["copies"]:
                src = self.pts.get((ctx, s), set())
                tgt = self.pts.setdefault((ctx, d), set())
                if not src <= tgt:
                    tgt |= src
                    wl.append((ctx, fn))
            for site, callee, args, retv in f["calls"]:
                nctx = self._ctx(ctx, site)
                cal = self.funcs[callee]
                # actuals -> formals
                for fm, ac in zip(cal["formals"], args, strict=True):
                    src = self.pts.get((ctx, ac), set())
                    tgt = self.pts.setdefault((nctx, fm), set())
                    if not src <= tgt:
                        tgt |= src
                        wl.append((nctx, callee))
                # return
                if cal["ret"] is not None:
                    rs = self.pts.get((nctx, cal["ret"]), set())
                    tgt = self.pts.setdefault((ctx, retv), set())
                    if not rs <= tgt:
                        tgt |= rs
                        wl.append((ctx, fn))
                # callee body may need a re-visit under this ctx anyway
                if (nctx, callee) not in wl:
                    wl.append((nctx, callee))

    def points_to(self, ctx: Ctx, var: str) -> set[str]:
        return set(self.pts.get((ctx, var), set()))

    def insensitive(self, var: str) -> set[str]:
        out: set[str] = set()
        for (_, v), s in self.pts.items():
            if v == var:
                out |= s
        return out


def bench_context_pta(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    a = CtxPTA(k=1)
    # id(x): return x
    a.add_fn("id", formals=["a"], copies=[("r", "a")], ret="r")
    # f: x=new o1; u=id(x);  g: y=new o2; v=id(y); main calls f(), g()
    a.add_fn("f", allocs=[("x", "o1")], calls=[("s1", "id", ["x"], "u")])
    a.add_fn("g", allocs=[("y", "o2")], calls=[("s2", "id", ["y"], "v")])
    a.add_fn("main", calls=[("c1", "f", [], "p"), ("c2", "g", [], "q")])
    a.solve("main")
    # formal 'a' under ctx s1 sees only &o1; under s2 only &o2
    checks.append(a.points_to(("s1",), "a") == {"&o1"})
    checks.append(a.points_to(("s2",), "a") == {"&o2"})
    # monovariant merge sees both
    checks.append(a.insensitive("a") == {"&o1", "&o2"})
    checks.append(a.points_to(("s1",), "r") == {"&o1"})
    # k=0 degenerates to context-insensitive
    b = CtxPTA(k=0)
    b.funcs = a.funcs
    b.solve("main")
    checks.append(b.points_to((), "a") == {"&o1", "&o2"})
    return {"synthetic_context_pta": float(sum(checks)) / len(checks)}
