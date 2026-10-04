"""Congruence closure for EUF — union-find over terms with signature merging.

Ground equations between first-order terms (atoms and function applications)
are decided by saturating congruence: if a = b then f(a) = f(b). Classes are
tracked by union-find; signatures (root symbol + representative ids of args)
give a deterministic merge order. Naive fixpoint — sound for small terms.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 999

Term = Any


def _is_app(t: Term) -> bool:
    return isinstance(t, tuple) and len(t) >= 1 and isinstance(t[0], str)


def _subterms(t: Term) -> list[Term]:
    out = [t]
    if _is_app(t):
        for a in t[1:]:
            out.extend(_subterms(a))
    return out


class Congruence:
    """Union-find with signature congruence, small-scale."""

    def __init__(self) -> None:
        self._parent: dict[Term, Term] = {}
        self._rank: dict[Term, int] = {}

    def add(self, t: Term) -> None:
        for s in _subterms(t):
            self._parent.setdefault(s, s)
            self._rank.setdefault(s, 0)

    def find(self, t: Term) -> Term:
        p = self._parent[t]
        if p != t:
            self._parent[t] = self.find(p)
        return self._parent[t]

    def signature(self, t: Term) -> tuple[Any, ...]:
        if not _is_app(t):
            return ("@", t)
        return (t[0],) + tuple(self.find(a) for a in t[1:])

    def merge(self, a: Term, b: Term) -> None:
        self.add(a)
        self.add(b)
        self._union(self.find(a), self.find(b))
        self._saturate()

    def _saturate(self) -> None:
        changed = True
        while changed:
            changed = False
            sigs: dict[tuple[Any, ...], Term] = {}
            for t in list(self._parent):
                sig = self.signature(t)
                if sig in sigs and self.find(sigs[sig]) != self.find(t):
                    self._union(self.find(sigs[sig]), self.find(t))
                    changed = True
                else:
                    sigs[sig] = t

    def _union(self, a: Term, b: Term) -> None:
        if self._rank[a] < self._rank[b]:
            a, b = b, a
        self._parent[b] = a
        if self._rank[a] == self._rank[b]:
            self._rank[a] += 1

    def equal(self, a: Term, b: Term) -> bool:
        self.add(a)
        self.add(b)
        self._saturate()
        return bool(self.find(a) == self.find(b))


def bench_congruence_closure(seed: int = _SEED) -> dict[str, float]:
    """EUF queries decided by saturated congruence."""
    del seed
    cc = Congruence()
    cc.merge("x", "y")
    checks: list[bool] = []
    checks.append(cc.equal(("f", "x"), ("f", "y")))
    checks.append(cc.equal(("f", ("f", "x")), ("f", ("f", "y"))))
    checks.append(not cc.equal(("f", "x"), ("g", "x")))

    cc2 = Congruence()
    cc2.merge("a", "b")
    cc2.merge("b", "c")
    cc2.merge(("f", "a"), "z")
    checks.append(cc2.equal(("f", "c"), "z"))
    checks.append(cc2.equal(("f", ("f", "c")), ("f", "z")))

    cc3 = Congruence()
    cc3.merge("x", "y")
    checks.append(not cc3.equal("x", "w"))
    return {"synthetic_congruence_closure": float(sum(checks)) / len(checks)}
