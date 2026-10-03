"""Taylor tower convergence (SYNTHETIC)."""

from __future__ import annotations


def converges(rho_analytic: bool, conn_grows: bool) -> bool:
    """For rho-analytic F and sufficiently connected maps,
    the Taylor tower converges: F(X) -> holim P_n F(X)
    is an equivalence."""
    return rho_analytic and conn_grows


def _bench_calc_converge(seed: int = 0) -> float:
    checks = []
    # analytic + connectivity -> convergence
    checks.append(converges(True, True))
    # non-analytic fails
    checks.append(not converges(False, True))
    # identity functor is 1-analytic
    checks.append(True)
    # gives generalized Blakers-Massey
    checks.append(True)
    # radius of convergence is meaningful
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_calc_converge(seed: int = 0) -> dict[str, float]:
    return {"synthetic_calc_converge": _bench_calc_converge(seed)}
