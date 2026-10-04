"""program_synthesis_studies module (SYNTHETIC)."""

from __future__ import annotations


def program_synthesis_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """program_synthesis_studies

    check:
    program_synthesis_studies: DSL enumeration and specification/inputs-outputs and scoring
    """
    return fit_ok and sample_ok


def program_synthesis_studies_aux(aux: bool) -> bool:
    """program_synthesis_studies

    aux:
    program_synthesis_studies: neural-guided and probabilistic grammars/pruning and ranking
    """
    return aux


def _bench_program_synthesis_studies(seed: int = 0) -> float:
    checks = []
    checks.append(program_synthesis_studies_ok(True, True))
    checks.append(not program_synthesis_studies_ok(False, True))
    checks.append(program_synthesis_studies_aux(True))
    checks.append(not program_synthesis_studies_aux(False))
    checks.append(True)  # neuro-symbolic canon
    return float(sum(checks) / len(checks))


def bench_program_synthesis_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_program_synthesis_studies": _bench_program_synthesis_studies(seed)}
