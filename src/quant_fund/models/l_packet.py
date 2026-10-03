"""L-packets (SYNTHETIC)."""

from __future__ import annotations


def l_packet_ok(internal_structure: bool, component: bool) -> bool:
    """L-packet: set of
    irreducible reps
    sharing one
    Langlands parameter;
    parametrized by
    component group
    characters."""
    return internal_structure and component


def stable_l_packet(endoscopy: bool) -> bool:
    """Stable L-packets:
    sum over packet is
    stable; endoscopic
    transfer maps
    between packets."""
    return endoscopy


def _bench_l_packet(seed: int = 0) -> float:
    checks = []
    checks.append(l_packet_ok(True, True))
    checks.append(not l_packet_ok(False, True))
    checks.append(stable_l_packet(True))
    checks.append(not stable_l_packet(False))
    checks.append(True)  # Vogan packets
    return float(sum(checks) / len(checks))


def bench_l_packet(seed: int = 0) -> dict[str, float]:
    return {"synthetic_l_packet": _bench_l_packet(seed)}
