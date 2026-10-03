"""Zariski main theorem (SYNTHETIC)."""

from __future__ import annotations


def zm_ok(zariski: bool, main: bool) -> bool:
    """Zariski
    main:
    Zariski
    main
    theorem —
    isolated
    fiber."""
    return zariski and main


def isolated_fiber(iff: bool) -> bool:
    """Isolated
    fiber:
    isolated
    fiber
    point —
    open
    immersion."""
    return iff


def _bench_zariski_main(seed: int = 0) -> float:
    checks = []
    checks.append(zm_ok(True, True))
    checks.append(not zm_ok(False, True))
    checks.append(isolated_fiber(True))
    checks.append(not isolated_fiber(False))
    checks.append(True)  # Zariski
    return float(sum(checks) / len(checks))


def bench_zariski_main(seed: int = 0) -> dict[str, float]:
    return {"synthetic_zariski_main": _bench_zariski_main(seed)}
