"""jailbreak_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def jailbreak_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jailbreak_bench_studies

    check:
    jailbreak_bench_studies: jailbreak prompts/attacks and bypass rates
    """
    return fit_ok and sample_ok


def jailbreak_bench_studies_aux(aux: bool) -> bool:
    """jailbreak_bench_studies

    aux:
    jailbreak_bench_studies: guardrail policies/threats and eval verdicts
    """
    return aux


def _bench_jailbreak_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jailbreak_bench_studies_ok(True, True))
    checks.append(not jailbreak_bench_studies_ok(False, True))
    checks.append(jailbreak_bench_studies_aux(True))
    checks.append(not jailbreak_bench_studies_aux(False))
    checks.append(True)  # safety-eval canon
    return float(sum(checks) / len(checks))


def bench_jailbreak_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jailbreak_bench_studies": _bench_jailbreak_bench_studies(seed)}
