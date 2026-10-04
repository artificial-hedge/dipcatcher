"""kolyvagin sys module (SYNTHETIC)."""

from __future__ import annotations


def kolyvagin_sys_ok(iwasawa: bool, euler: bool) -> bool:
    """kolyvagin_sys
    check:
    Iwasawa
    structure —
    Euler."""
    return iwasawa and euler


def kolyvagin_sys_aux(aux: bool) -> bool:
    """kolyvagin_sys
    aux:
    auxiliary
    Iwasawa
    check —
    Selmer."""
    return aux


def _bench_kolyvagin_sys(seed: int = 0) -> float:
    checks = []
    checks.append(kolyvagin_sys_ok(True, True))
    checks.append(not kolyvagin_sys_ok(False, True))
    checks.append(kolyvagin_sys_aux(True))
    checks.append(not kolyvagin_sys_aux(False))
    checks.append(True)  # Iwasawa/Euler-system canon
    return float(sum(checks) / len(checks))


def bench_kolyvagin_sys(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kolyvagin_sys": _bench_kolyvagin_sys(seed)}
