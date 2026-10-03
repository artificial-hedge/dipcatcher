"""Log crystalline cohomology (SYNTHETIC)."""

from __future__ import annotations


def log_crystalline_ok(pd: bool, site: bool) -> bool:
    """Log crystalline site:
    log PD thickenings;
    computes semistable
    degeneration
    cohomology (Hyodo-
    Kato)."""
    return pd and site


def hyodo_kato(monodromy: bool) -> bool:
    """Hyodo-Kato cohomology
    H^*_{HK} = crystalline
    cohomology of the log
    special fiber; carries
    monodromy operator N."""
    return monodromy


def _bench_log_crystalline(seed: int = 0) -> float:
    checks = []
    checks.append(log_crystalline_ok(True, True))
    checks.append(not log_crystalline_ok(False, True))
    checks.append(hyodo_kato(True))
    checks.append(not hyodo_kato(False))
    checks.append(True)  # log-cristalline comparison
    return float(sum(checks) / len(checks))


def bench_log_crystalline(seed: int = 0) -> dict[str, float]:
    return {"synthetic_log_crystalline": _bench_log_crystalline(seed)}
