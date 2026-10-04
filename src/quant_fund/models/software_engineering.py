"""software_engineering module (SYNTHETIC)."""

from __future__ import annotations


def software_engineering_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """software_engineering

    check:
    computer_science_2: computer science
    software_engineering: software engineering
    machine_learning_2: machine learning
    artificial_intelligence: artificial intelligence
    data_engineering: data engineering
    information_theory_2: information theory
    """
    return fit_ok and sample_ok


def software_engineering_aux(aux: bool) -> bool:
    """software_engineering

    aux:
    computer_science_2: algorithms and complexity
    software_engineering: builds and deployments
    machine_learning_2: gradients and losses
    artificial_intelligence: agents and reasoning
    data_engineering: ingestion and schemas
    information_theory_2: entropy and codes
    """
    return aux


def _bench_software_engineering(seed: int = 0) -> float:
    checks = []
    checks.append(software_engineering_ok(True, True))
    checks.append(not software_engineering_ok(False, True))
    checks.append(software_engineering_aux(True))
    checks.append(not software_engineering_aux(False))
    checks.append(True)  # computing-sciences canon
    return float(sum(checks) / len(checks))


def bench_software_engineering(seed: int = 0) -> dict[str, float]:
    return {"synthetic_software_engineering": _bench_software_engineering(seed)}
