"""Loop space / suspension adjunction bookkeeping (SYNTHETIC)."""

from __future__ import annotations


def loop_em(group: str, n: int) -> tuple[str, int]:
    """Omega K(G, n) = K(G, n-1): loops shift the Eilenberg-MacLane index."""
    return (group, n - 1)


def suspension_em(group: str, n: int) -> tuple[str, int]:
    """Suspension of the classifying picture raises cohomological degree."""
    return (group, n + 1)


def freudenthal_iso(n: int, i: int) -> bool:
    """pi_i(S^n) -> pi_{i+1}(S^{n+1}) is an isomorphism for i <= 2n - 2."""
    return i <= 2 * n - 2


def _bench_loop_space(seed: int = 0) -> float:
    checks = []
    # Omega K(Z,3) = K(Z,2); Omega^2 K(Z,3) = K(Z,1)
    checks.append(loop_em("Z", 3) == ("Z", 2))
    checks.append(loop_em(*loop_em("Z", 3)) == ("Z", 1))
    # loop of the circle's universal-cover path object lands on K(Z,0)=Z
    checks.append(loop_em("Z", 1) == ("Z", 0))
    # suspension of K(Z,1)=S1 picture gives K(Z,2)=CP^inf level
    checks.append(suspension_em("Z", 1) == ("Z", 2))
    # Freudenthal: pi_3(S^2) -> pi_4(S^3) iso at i=3 <= 2*2-2=2? no: i=3>2
    checks.append(not freudenthal_iso(2, 3))
    # pi_2(S^2) -> pi_3(S^3): i=2 <= 2 -> iso (2*2-2=2 boundary)
    checks.append(freudenthal_iso(2, 2))
    # pi_1(S^1) -> pi_2(S^2) is stable iso (1 <= 0? boundary n=1: i<=0 no)
    checks.append(not freudenthal_iso(1, 1))
    return float(sum(checks) / len(checks))


def bench_loop_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_loop_space": _bench_loop_space(seed)}
