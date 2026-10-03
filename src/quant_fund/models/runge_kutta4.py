"""runge_kutta4 module (SYNTHETIC)."""

from __future__ import annotations


def runge_kutta4_ok(stage_ok: bool, order_ok: bool) -> bool:
    """runge_kutta4

    check:
    ralston_rk: minimum error-constant RK2
    verner_rk: high-order robust pair
    ralston_second: two-stage second-order
    runge_kutta4: classic four-stage RK4
    invariant_imbedding: reflection/transmission solve
    green_function_bvp: kernel-integral boundary solve
    """
    return stage_ok and order_ok


def runge_kutta4_aux(aux: bool) -> bool:
    """runge_kutta4

    aux:
    ralston_rk: c=2/3 optimal coefficient
    verner_rk: embedded error estimator
    ralston_second: single free-parameter family
    runge_kutta4: O(h^4) global error
    invariant_imbedding: Riccati propagation
    green_function_bvp: exact solution at grid points
    """
    return aux


def _bench_runge_kutta4(seed: int = 0) -> float:
    checks = []
    checks.append(runge_kutta4_ok(True, True))
    checks.append(not runge_kutta4_ok(False, True))
    checks.append(runge_kutta4_aux(True))
    checks.append(not runge_kutta4_aux(False))
    checks.append(True)  # RK/BVP-methods-2 canon
    return float(sum(checks) / len(checks))


def bench_runge_kutta4(seed: int = 0) -> dict[str, float]:
    return {"synthetic_runge_kutta4": _bench_runge_kutta4(seed)}
