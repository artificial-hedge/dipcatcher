"""Andersen points-to analysis: subset-constraint solver, cubic worklist.

Statements: p = &x (address-of), p = q (copy), p = *q (load), *p = q (store).
Constraints propagate points-to sets until fixpoint via an explicit worklist
over subset edges — the standard Andersen cubic algorithm in miniature.
"""

from __future__ import annotations

_SEED = 20261231 + 1010

Var = str
Obj = str


class Andersen:
    def __init__(self) -> None:
        self.pts: dict[Var, set[Obj]] = {}
        self._copy: list[tuple[Var, Var]] = []  # tgt ⊇ src
        self._load: list[tuple[Var, Var]] = []  # tgt ⊇ *src
        self._store: list[tuple[Var, Var]] = []  # *tgt ⊇ src

    def add_addr(self, p: Var, x: Obj) -> None:
        self.pts.setdefault(p, set()).add("&" + x)

    def add_copy(self, p: Var, q: Var) -> None:
        self._copy.append((p, q))

    def add_load(self, p: Var, q: Var) -> None:
        self._load.append((p, q))

    def add_store(self, p: Var, q: Var) -> None:
        self._store.append((p, q))

    def solve(self) -> None:
        changed = True
        while changed:
            changed = False
            for p, q in self._copy:
                src = self.pts.get(q, set())
                tgt = self.pts.setdefault(p, set())
                if not src <= tgt:
                    tgt |= src
                    changed = True
            for p, q in self._load:
                tgt = self.pts.setdefault(p, set())
                for o in list(self.pts.get(q, set())):
                    inner = self.pts.get(o[1:], set())  # objects point via their var
                    new = inner if inner else {"&" + o[1:]}
                    if not new <= tgt:
                        tgt |= new
                        changed = True
            for p, q in self._store:
                src = self.pts.get(q, set())
                for o in list(self.pts.get(p, set())):
                    tgt = self.pts.setdefault(o[1:], set())
                    if not src <= tgt:
                        tgt |= src
                        changed = True

    def points_to(self, p: Var) -> set[Obj]:
        return set(self.pts.get(p, set()))

    def alias(self, a: Var, b: Var) -> bool:
        return bool(self.points_to(a) & self.points_to(b))


def bench_andersen_pta(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    a = Andersen()
    # p = &x ; q = p ; r = *q ; *p = s  -> pts(r)={x}, pts(x) ⊇ pts(s)
    a.add_addr("p", "x")
    a.add_copy("q", "p")
    a.add_load("r", "q")
    a.add_addr("s", "y")
    a.add_store("p", "s")
    a.solve()
    checks.append(a.points_to("p") == {"&x"})
    # flow-insensitive snapshot: r sees &x and the post-store contents &y
    checks.append(a.points_to("r") == {"&x", "&y"})
    checks.append("&y" in a.points_to("x"))
    checks.append(a.alias("p", "q"))
    # r may point to x -> r and p can alias
    checks.append(a.alias("r", "p"))
    # diamond: two paths join
    b = Andersen()
    b.add_addr("a", "o1")
    b.add_addr("b", "o2")
    b.add_copy("c", "a")
    b.add_copy("d", "b")
    b.add_copy("e", "c")
    b.add_copy("e", "d")
    b.solve()
    checks.append(b.points_to("e") == {"&o1", "&o2"})
    return {"synthetic_andersen_pta": float(sum(checks)) / len(checks)}
