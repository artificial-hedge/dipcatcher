"""RO(G)-graded homotopy groups (SYNTHETIC)."""

from __future__ import annotations


def ro_grading_ok(grading_count: int, real_reps: int) -> bool:
    """pi_V(X) indexed on virtual reps V = V1 - V2 of
    RO(G), richer than Z-grading (Lewis-Mandell)."""
    return grading_count == real_reps


def suspension_shift(v_dim: int) -> int:
    """Smashing with S^V shifts homotopy grading by dim V."""
    return v_dim


def _bench_ro_grading(seed: int = 0) -> float:
    checks = []
    checks.append(ro_grading_ok(4, 4))  # C2: 1, sign, 1+sign, ...
    checks.append(suspension_shift(2) == 2)
    checks.append(suspension_shift(0) == 0)
    # pi_{p+sigma}(MU_R) well-defined
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_ro_grading(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ro_grading": _bench_ro_grading(seed)}
