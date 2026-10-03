"""Resolution proof-size bounds on toy formulas (SYNTHETIC)."""

from __future__ import annotations

from itertools import combinations


def resolve(c1: frozenset[str], c2: frozenset[str]) -> frozenset[str] | None:
    """Resolution on literal-clauses (strings with leading ~ for neg)."""
    for lit in c1:
        comp = lit[1:] if lit.startswith("~") else "~" + lit
        if comp in c2:
            return frozenset((c1 - {lit}) | (c2 - {comp}))
    return None


def php_clauses(holes: int, pigeons: int) -> list[frozenset[str]]:
    """PHP_n^{n+1}: pigeon i goes to some hole; no two pigeons share a
    hole. Literals: 'i.h' = pigeon i in hole h."""
    cls = []
    for i in range(pigeons):
        cls.append(frozenset(f"{i}.{h}" for h in range(holes)))
    for i, j in combinations(range(pigeons), 2):
        for h in range(holes):
            cls.append(frozenset({f"~{i}.{h}", f"~{j}.{h}"}))
    return cls


def refutable(clauses: list[frozenset[str]], fuel: int = 1000) -> bool:
    """SAT-check via exhaustive resolution closure (bounded)."""
    work = set(clauses)
    step = 0
    changed = True
    while changed and step < fuel:
        changed = False
        step += 1
        for c1, c2 in combinations(sorted(work, key=len), 2):
            r = resolve(c1, c2)
            if r is not None and r not in work:
                if not r:
                    return True
                work.add(r)
                changed = True
    return frozenset() in work


def _bench_proof_complexity(seed: int = 0) -> float:
    checks = []
    # PHP_2^3 (3 pigeons, 2 holes) is unsatisfiable
    checks.append(refutable(php_clauses(2, 3)))
    # PHP_2^2 is satisfiable: no empty clause derivable
    checks.append(not refutable(php_clauses(2, 2)))
    # unit propagation resolves a 2-clause clash
    checks.append(resolve(frozenset({"a"}), frozenset({"~a"})) == frozenset())
    checks.append(resolve(frozenset({"a", "b"}), frozenset({"~a"})) == frozenset({"b"}))
    # PHP_1^2 (2 pigeons, 1 hole) unsat: clauses {1.1},{2.1},{~1.1,~2.1}
    checks.append(refutable(php_clauses(1, 2)))
    # clause count: PHP_2^3 = 3 pigeon + 3 holes*C(3,2)=9 clash = 12
    checks.append(len(php_clauses(2, 3)) == 9)
    return float(sum(checks) / len(checks))


def bench_proof_complexity(seed: int = 0) -> dict[str, float]:
    return {"synthetic_proof_complexity": _bench_proof_complexity(seed)}
