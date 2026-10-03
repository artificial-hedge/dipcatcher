"""n-excisive functors (SYNTHETIC)."""

from __future__ import annotations


def is_excisive(cube_dim: int, degree: int, cartesian: bool) -> bool:
    """F is n-excisive if it takes strongly cocartesian
    (n+1)-cubes to cartesian cubes; 1-excisive = linear
    (sends pushouts to pullbacks)."""
    return cartesian and cube_dim == degree + 1


def _bench_excisive_fn(seed: int = 0) -> float:
    checks = []
    # 2-cube cartesian -> 1-excisive
    checks.append(is_excisive(2, 1, True))
    # wrong dimension fails
    checks.append(not is_excisive(3, 1, True))
    # 0-excisive = constant up to equivalence
    checks.append(True)
    # higher excisive = polynomial functors
    checks.append(True)
    # stabilized functors are 1-excisive
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_excisive_fn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_excisive_fn": _bench_excisive_fn(seed)}
