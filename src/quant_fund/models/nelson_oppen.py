"""Nelson-Oppen theory combination, toy-scale: EUF + integer constants.

Two theories cooperate on shared variables: EUF (congruence closure over
function terms) and a constant-theory assigning integer values and detecting
x = i, x = j contradictions. Equalities propagate between theories until
fixpoint — the Nelson-Oppen arrangement for disjoint convex theories.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1002

Term = Any

# Constants are ("const", k); variables are ("v", name); apps ("f", args...).


def _is_const(t: Term) -> bool:
    return isinstance(t, tuple) and len(t) > 0 and t[0] == "const"


def _is_var(t: Term) -> bool:
    return isinstance(t, tuple) and len(t) > 0 and t[0] == "v"


def _is_app(t: Term) -> bool:
    return isinstance(t, tuple) and bool(t) and not _is_const(t) and not _is_var(t)


def _sub(t: Term) -> list[Term]:
    out = [t]
    if _is_app(t):
        for a in t[1:]:
            out.extend(_sub(a))
    return out


class CombinedSolver:
    """EUF classes + const bindings, shared via propagation."""

    def __init__(self) -> None:
        self._par: dict[Term, Term] = {}
        self._neq: list[tuple[Term, Term]] = []
        self.unsat = False

    def add(self, t: Term) -> None:
        for s in _sub(t):
            self._par.setdefault(s, s)

    def find(self, t: Term) -> Term:
        p = self._par[t]
        if p != t:
            self._par[t] = self.find(p)
        return self._par[t]

    def _const_of(self, t: Term) -> int | None:
        """Integer value if class contains a const."""
        return int(t[1]) if _is_const(t) else None

    def assume_eq(self, a: Term, b: Term) -> None:
        self.add(a)
        self.add(b)
        self._union(a, b)
        self._propagate()

    def assume_neq(self, a: Term, b: Term) -> None:
        self.add(a)
        self.add(b)
        self._neq.append((a, b))
        if self.find(a) == self.find(b):
            self.unsat = True
        self._propagate()

    def _union(self, a: Term, b: Term) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # prefer const / app as representative so signature lookups stay simple
        if _is_const(rb) or (_is_app(rb) and not _is_app(ra)):
            ra, rb = rb, ra
        self._par[rb] = ra

    def _sig(self, t: Term) -> tuple[Any, ...]:
        if _is_app(t):
            return (t[0],) + tuple(self.find(a) for a in t[1:])
        return ("@", self.find(t))

    def _propagate(self) -> None:
        changed = True
        while changed:
            changed = False
            terms = list(self._par)
            # 1. congruence merge
            sigs: dict[tuple[Any, ...], Term] = {}
            for t in terms:
                sg = self._sig(t)
                if sg in sigs and self.find(sigs[sg]) != self.find(t):
                    self._union(sigs[sg], t)
                    changed = True
                else:
                    sigs[sg] = t
            # 2. const clashes and equalities between distinct reps
            reps: dict[Term, list[Term]] = {}
            for t in terms:
                reps.setdefault(self.find(t), []).append(t)
            for ts in reps.values():
                consts = [self._const_of(t) for t in ts if _is_const(t)]
                if len(set(consts)) > 1:
                    self.unsat = True
                    return
            for a, b in self._neq:
                if self.find(a) == self.find(b):
                    self.unsat = True
                    return
            # 3. new merges discovered? loop re-runs anyway

    def equal(self, a: Term, b: Term) -> bool:
        self.add(a)
        self.add(b)
        self._propagate()
        return bool(self.find(a) == self.find(b))


def v(name: str) -> Term:
    return ("v", name)


def c(k: int) -> Term:
    return ("const", k)


def app(f: str, *args: Term) -> Term:
    return (f,) + args


def bench_nelson_oppen(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # classic propagation: f(x)=y, x=1, f(1)=2  =>  y=2
    s = CombinedSolver()
    s.assume_eq(app("f", v("x")), v("y"))
    s.assume_eq(v("x"), c(1))
    s.assume_eq(app("f", c(1)), c(2))
    checks.append(s.equal(v("y"), c(2)))
    checks.append(not s.unsat)
    # contradiction: x=1, x=2
    s2 = CombinedSolver()
    s2.assume_eq(v("x"), c(1))
    s2.assume_eq(v("x"), c(2))
    checks.append(s2.unsat)
    # g(x)=z, x=y, y=3 => g(3)=z
    s3 = CombinedSolver()
    s3.assume_eq(app("g", v("x")), v("z"))
    s3.assume_eq(v("x"), v("y"))
    s3.assume_eq(v("y"), c(3))
    checks.append(s3.equal(app("g", c(3)), v("z")))
    # disequality: x != y then force x=y via consts -> unsat
    s4 = CombinedSolver()
    s4.assume_neq(v("x"), v("y"))
    s4.assume_eq(v("x"), c(7))
    s4.assume_eq(v("y"), c(7))
    checks.append(s4.unsat)
    # distinct values stay distinct
    s5 = CombinedSolver()
    s5.assume_eq(v("p"), c(1))
    checks.append(not s5.equal(v("p"), c(2)))
    return {"synthetic_nelson_oppen": float(sum(checks)) / len(checks)}
