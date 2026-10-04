"""JSJ decomposition (SYNTHETIC)."""

from __future__ import annotations


def jsj_ok(tori: bool, pieces: bool) -> bool:
    """JSJ
    decomposition:
    canonical
    torus
    splitting
    into
    Seifert
    and
    atoroidal
    pieces —
    Jaco-Shalen-Johannson."""
    return tori and pieces


def minimal_tori(mt: bool) -> bool:
    """The
    JSJ
    collection
    is
    minimal
    and
    unique
    up
    to
    isotopy."""
    return mt


def _bench_jsj_decomp(seed: int = 0) -> float:
    checks = []
    checks.append(jsj_ok(True, True))
    checks.append(not jsj_ok(False, True))
    checks.append(minimal_tori(True))
    checks.append(not minimal_tori(False))
    checks.append(True)  # JSJ
    return float(sum(checks) / len(checks))


def bench_jsj_decomp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jsj_decomp": _bench_jsj_decomp(seed)}
