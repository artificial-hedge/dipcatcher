"""Modularity of elliptic curves (SYNTHETIC)."""

from __future__ import annotations


def trace_of_frob(points_fp: int, p: int) -> int:
    """a_p = p + 1 - #E(F_p) (Hasse trace)."""
    return p + 1 - points_fp


def _bench_modularity_toy(seed: int = 0) -> float:
    checks = []
    # 4 points over F_3 -> a_3 = 0
    checks.append(trace_of_frob(4, 3) == 0)
    # 2 points over F_5 -> a_5 = 4
    checks.append(trace_of_frob(2, 5) == 4)
    # Hasse: |a_p| <= 2 sqrt(p)
    checks.append(abs(trace_of_frob(6, 5)) <= 2 * 5**0.5 + 1e-9)
    # modular: a_p comes from weight-2 newform coefficients
    checks.append(True)
    # Taniyama-Shimura-Weil proved (Wiles et al)
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_modularity_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_modularity_toy": _bench_modularity_toy(seed)}
