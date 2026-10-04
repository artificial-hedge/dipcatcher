"""Profunctor (distributor) composition on finite sets (SYNTHETIC)."""

from __future__ import annotations


def prof_compose(phi: set[tuple[str, str]], psi: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """(Phi . Psi)(a, c) iff exists b with (a,b) in Phi, (b,c) in Psi."""
    return {(a, c) for (a, b) in phi for (b2, c) in psi if b2 == b}


def prof_unit(xs: set[str]) -> set[tuple[str, str]]:
    """Unit distributor = diagonal/hom relation."""
    return {(x, x) for x in xs}


def _bench_distributor(seed: int = 0) -> float:
    checks = []
    phi = {("a", "x"), ("a", "y"), ("b", "x")}
    psi = {("x", "p"), ("y", "q")}
    comp = prof_compose(phi, psi)
    checks.append(comp == {("a", "p"), ("a", "q"), ("b", "p")})
    # unit laws
    u = prof_unit({"a", "b"})
    checks.append(prof_compose(u, phi) == phi)
    checks.append(prof_compose(phi, prof_unit({"x", "y"})) == phi)
    # associativity on a third relation
    chi = {("p", "m"), ("q", "n")}
    checks.append(
        prof_compose(prof_compose(phi, psi), chi) == prof_compose(phi, prof_compose(psi, chi))
    )
    # empty distributor annihilates
    checks.append(prof_compose(set(), psi) == set())
    return float(sum(checks) / len(checks))


def bench_distributor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_distributor": _bench_distributor(seed)}
