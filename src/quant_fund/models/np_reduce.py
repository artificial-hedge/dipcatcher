"""NP-completeness reduction gadgets + verification vs brute force (SYNTHETIC bench)."""

from __future__ import annotations

import itertools

Clause = tuple[int, ...]  # signed literals: +i means x_i true


def sat_to_3sat(clauses: list[Clause], n_vars: int) -> tuple[list[Clause], int]:
    """Tseitin-style splitting: every clause becomes length<=3 (equisatisfiable)."""
    out: list[Clause] = []
    fresh = n_vars
    for c0 in clauses:
        c: list[int] = list(c0)
        while len(c) > 3:
            fresh += 1
            a, b = c[0], c[1]
            out.append((a, b, fresh))
            c = [-(fresh)] + c[2:]
        out.append(tuple(c))
    return out, fresh


def eval_clause(c: Clause, assign: dict[int, bool]) -> bool:
    return any(assign.get(abs(lit)) == (lit > 0) for lit in c)


def brute_sat(clauses: list[Clause], n_vars: int) -> bool:
    for bits in itertools.product([False, True], repeat=n_vars):
        assign = {i + 1: b for i, b in enumerate(bits)}
        if all(eval_clause(c, assign) for c in clauses):
            return True
    return False


def three_sat_to_vc(clauses: list[Clause], n_vars: int) -> tuple[list[tuple[int, int]], int, int]:
    """3SAT -> VertexCover: var-gadgets + clause-triangles + consistency edges.

    Nodes: 0..2n-1 are literal nodes (2i = x_i, 2i+1 = ¬x_i for i in 1..n;
    use id 2*(v-1)/2*(v-1)+1); clause nodes start at 2n.
    Edges: (xi,¬xi) per var; triangle per clause; literal node of clause
    connects to the same literal var node.
    k = n_vars + 2 * n_clauses.
    """

    def varnode(lit: int) -> int:
        v = abs(lit)
        return 2 * (v - 1) + (1 if lit < 0 else 0)

    edges: list[tuple[int, int]] = []
    for v in range(1, n_vars + 1):
        edges.append((2 * (v - 1), 2 * (v - 1) + 1))
    base = 2 * n_vars
    for ci, c in enumerate(clauses):
        tri = [base + 3 * ci + k for k in range(3)]
        edges += [(tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])]
        for k, lit in enumerate(c):
            edges.append((tri[k], varnode(lit)))
    n_nodes = base + 3 * len(clauses)
    k = n_vars + 2 * len(clauses)
    return edges, k, n_nodes


def brute_vc(edges: list[tuple[int, int]], n_nodes: int, k: int) -> bool:
    for comb in itertools.combinations(range(n_nodes), k):
        s = set(comb)
        if all(u in s or v in s for u, v in edges):
            return True
    return False


def _bench_np_reduce(seed: int = 0) -> float:
    import random

    rng = random.Random(20261231 + 1066)
    checks = []
    # equisatisfiability of the 3SAT transform over random formulas
    for _ in range(6):
        nv = 4
        clauses = [
            tuple(rng.sample(range(-nv, nv + 1), rng.choice([2, 3, 4, 5]))) for _ in range(4)
        ]
        clauses = [tuple(lit for lit in c if lit != 0) for c in clauses]
        c3, n3 = sat_to_3sat(clauses, nv)
        checks.append(all(len(c) <= 3 for c in c3))
        checks.append(brute_sat(clauses, nv) == brute_sat(c3, n3))
    # small VC gadget check: x1 \/ x2 \/ x3 satisfiable -> VC of size k exists
    edges, k, nn = three_sat_to_vc([(1, 2, 3)], 3)
    checks.append(brute_vc(edges, nn, k))
    # unsat single-var formula: (x1),(¬x1) — but 3SAT transform keeps 1-clauses;
    # use (x1∨x1∨x1),(¬x1∨¬x1∨¬x1): unsat -> no VC of size k
    edges2, k2, nn2 = three_sat_to_vc([(1, 1, 1), (-1, -1, -1)], 1)
    checks.append(not brute_vc(edges2, nn2, k2))
    return sum(checks) / len(checks)


def bench_np_reduce(seed: int = 0) -> dict[str, float]:
    return {"synthetic_np_reduce": _bench_np_reduce(seed)}
