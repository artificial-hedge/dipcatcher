"""logic_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def logic_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """logic_bench_studies

    check:
    logic_bench_studies: LogicBench rule-following inference accuracy
    """
    return fit_ok and sample_ok


def logic_bench_studies_aux(aux: bool) -> bool:
    """logic_bench_studies

    aux:
    logic_bench_studies: premises, conclusions, and correct rates
    """
    return aux


def _bench_logic_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(logic_bench_studies_ok(True, True))
    checks.append(not logic_bench_studies_ok(False, True))
    checks.append(logic_bench_studies_aux(True))
    checks.append(not logic_bench_studies_aux(False))
    checks.append(True)  # reasoning-eval canon
    return float(sum(checks) / len(checks))


def bench_logic_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_logic_bench_studies": _bench_logic_bench_studies(seed)}
