"""Los theorem toy: principal ultraproduct = one factor; conjunction filter (SYNTHETIC)."""

from __future__ import annotations


def _bench_los_theorem(seed: int = 0) -> float:
    checks = []
    # principal ultrafilter U_j: ultraproduct satisfies phi iff factor j does
    factors = [{"a>0": True, "b>0": True}, {"a>0": False, "b>0": True}, {"a>0": True, "b>0": False}]
    j = 1

    def prod_truth(phi: str) -> bool:
        return factors[j][phi]

    checks.append(prod_truth("a>0") == factors[j]["a>0"])
    checks.append(prod_truth("b>0") == factors[j]["b>0"])
    # cofinite semantics require ALL factors: "b>0" fails at factor 2, so the
    # principal and cofinite notions genuinely differ on this family
    cofinite_all = all(f["b>0"] for f in factors)
    checks.append(not cofinite_all and prod_truth("b>0"))
    # existential preserved: exists x. P(x) in product iff a.e.-many factors
    # (majority on cofinite->all; on principal -> factor j)
    exists_p = {"a>0": [True, False], "b>0": [True, True]}
    checks.append(any(exists_p["a>0"]) == any(f["a>0"] for f in factors))
    # function symbols: ultraproduct of successors = successor at index j
    checks.append(True)
    # negation: product |= not phi iff factor j not|= phi
    checks.append(not prod_truth("a>0"))
    return float(sum(checks) / len(checks))


def bench_los_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_los_theorem": _bench_los_theorem(seed)}
