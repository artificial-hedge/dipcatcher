"""de_novo_design_studies module (SYNTHETIC)."""

from __future__ import annotations


def de_novo_design_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """de_novo_design_studies

    check:
    de_novo_design_studies: generation and validity/novelty and design
    """
    return fit_ok and sample_ok


def de_novo_design_studies_aux(aux: bool) -> bool:
    """de_novo_design_studies

    aux:
    de_novo_design_studies: reward and fragment/property and synthesizability
    """
    return aux


def _bench_de_novo_design_studies(seed: int = 0) -> float:
    checks = []
    checks.append(de_novo_design_studies_ok(True, True))
    checks.append(not de_novo_design_studies_ok(False, True))
    checks.append(de_novo_design_studies_aux(True))
    checks.append(not de_novo_design_studies_aux(False))
    checks.append(True)  # drug-discovery canon
    return float(sum(checks) / len(checks))


def bench_de_novo_design_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_de_novo_design_studies": _bench_de_novo_design_studies(seed)}
