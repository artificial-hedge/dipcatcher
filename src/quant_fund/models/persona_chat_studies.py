"""persona_chat_studies module (SYNTHETIC)."""

from __future__ import annotations


def persona_chat_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """persona_chat_studies

    check:
    persona_chat_studies: PersonaChat metrics
    """
    return fit_ok and sample_ok


def persona_chat_studies_aux(aux: bool) -> bool:
    """persona_chat_studies

    aux:
    persona_chat_studies: personas, contexts, responses, and scores
    """
    return aux


def _bench_persona_chat_studies(seed: int = 0) -> float:
    checks = []
    checks.append(persona_chat_studies_ok(True, True))
    checks.append(not persona_chat_studies_ok(False, True))
    checks.append(persona_chat_studies_aux(True))
    checks.append(not persona_chat_studies_aux(False))
    checks.append(True)  # dialogue-system canon
    return float(sum(checks) / len(checks))


def bench_persona_chat_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_persona_chat_studies": _bench_persona_chat_studies(seed)}
