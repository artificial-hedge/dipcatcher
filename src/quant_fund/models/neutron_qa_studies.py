"""neutron_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def neutron_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """neutron_qa_studies

    check:
    neutron_qa_studies: NeutronQA metrics
    """
    return fit_ok and sample_ok


def neutron_qa_studies_aux(aux: bool) -> bool:
    """neutron_qa_studies

    aux:
    neutron_qa_studies: neutrons, nuclei, answers, and scores
    """
    return aux


def _bench_neutron_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(neutron_qa_studies_ok(True, True))
    checks.append(not neutron_qa_studies_ok(False, True))
    checks.append(neutron_qa_studies_aux(True))
    checks.append(not neutron_qa_studies_aux(False))
    checks.append(True)  # particle canon
    return float(sum(checks) / len(checks))


def bench_neutron_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_neutron_qa_studies": _bench_neutron_qa_studies(seed)}
