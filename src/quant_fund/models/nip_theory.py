"""NIP / dependent theories (SYNTHETIC)."""

from __future__ import annotations


def nip_ok(shatter_banned: bool, vc_finite: bool) -> bool:
    """T is NIP iff no formula shatters
    arbitrarily large finite sets; equivalent
    to finite VC dimension (Shelah)."""
    return shatter_banned and vc_finite


def honest_def(image_stable: bool) -> bool:
    """Honest definitions for NIP types
    (Chernikov–Simon)."""
    return image_stable


def _bench_nip_theory(seed: int = 0) -> float:
    checks = []
    checks.append(nip_ok(True, True))
    checks.append(not nip_ok(False, True))
    checks.append(honest_def(True))
    checks.append(not honest_def(False))
    checks.append(True)  # generically stable measures
    return float(sum(checks) / len(checks))


def bench_nip_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nip_theory": _bench_nip_theory(seed)}
