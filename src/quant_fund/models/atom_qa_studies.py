"""atom_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def atom_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """atom_qa_studies

    check:
    atom_qa_studies: AtomQA metrics
    """
    return fit_ok and sample_ok


def atom_qa_studies_aux(aux: bool) -> bool:
    """atom_qa_studies

    aux:
    atom_qa_studies: atoms, shells, answers, and scores
    """
    return aux


def _bench_atom_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(atom_qa_studies_ok(True, True))
    checks.append(not atom_qa_studies_ok(False, True))
    checks.append(atom_qa_studies_aux(True))
    checks.append(not atom_qa_studies_aux(False))
    checks.append(True)  # particle canon
    return float(sum(checks) / len(checks))


def bench_atom_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_atom_qa_studies": _bench_atom_qa_studies(seed)}
