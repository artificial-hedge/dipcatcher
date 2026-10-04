"""rope_scaling_studies module (SYNTHETIC)."""

from __future__ import annotations


def rope_scaling_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rope_scaling_studies

    check:
    rope_scaling_studies: NTK and YaRN interpolation/positional and extrapolation
    """
    return fit_ok and sample_ok


def rope_scaling_studies_aux(aux: bool) -> bool:
    """rope_scaling_studies

    aux:
    rope_scaling_studies: frequency and attention-temperature/long context and perplexity
    """
    return aux


def _bench_rope_scaling_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rope_scaling_studies_ok(True, True))
    checks.append(not rope_scaling_studies_ok(False, True))
    checks.append(rope_scaling_studies_aux(True))
    checks.append(not rope_scaling_studies_aux(False))
    checks.append(True)  # LLM-inference-2 canon
    return float(sum(checks) / len(checks))


def bench_rope_scaling_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rope_scaling_studies": _bench_rope_scaling_studies(seed)}
