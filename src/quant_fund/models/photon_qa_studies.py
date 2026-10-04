"""photon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def photon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """photon_qa_studies

    check:
    photon_qa_studies: PhotonQA metrics
    """
    return fit_ok and sample_ok


def photon_qa_studies_aux(aux: bool) -> bool:
    """photon_qa_studies

    aux:
    photon_qa_studies: photons, spectra, answers, and scores
    """
    return aux


def _bench_photon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(photon_qa_studies_ok(True, True))
    checks.append(not photon_qa_studies_ok(False, True))
    checks.append(photon_qa_studies_aux(True))
    checks.append(not photon_qa_studies_aux(False))
    checks.append(True)  # particle canon
    return float(sum(checks) / len(checks))


def bench_photon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_photon_qa_studies": _bench_photon_qa_studies(seed)}
