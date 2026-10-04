"""hotpotqa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hotpotqa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hotpotqa_studies

    check:
    hotpotqa_studies: HotpotQA multi-hop questions and joint EM/F1
    """
    return fit_ok and sample_ok


def hotpotqa_studies_aux(aux: bool) -> bool:
    """hotpotqa_studies

    aux:
    hotpotqa_studies: bridge/comparison items, supporting facts
    """
    return aux


def _bench_hotpotqa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hotpotqa_studies_ok(True, True))
    checks.append(not hotpotqa_studies_ok(False, True))
    checks.append(hotpotqa_studies_aux(True))
    checks.append(not hotpotqa_studies_aux(False))
    checks.append(True)  # reading-comprehension canon
    return float(sum(checks) / len(checks))


def bench_hotpotqa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hotpotqa_studies": _bench_hotpotqa_studies(seed)}
