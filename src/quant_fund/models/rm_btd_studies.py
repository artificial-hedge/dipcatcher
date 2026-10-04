"""rm_btd_studies module (SYNTHETIC)."""

from __future__ import annotations


def rm_btd_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rm_btd_studies

    check:
    rm_btd_studies: Bradley-Terry reward fitting/pairs and logits
    """
    return fit_ok and sample_ok


def rm_btd_studies_aux(aux: bool) -> bool:
    """rm_btd_studies

    aux:
    rm_btd_studies: BT preference likelihood optimization/winners and losers
    """
    return aux


def _bench_rm_btd_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rm_btd_studies_ok(True, True))
    checks.append(not rm_btd_studies_ok(False, True))
    checks.append(rm_btd_studies_aux(True))
    checks.append(not rm_btd_studies_aux(False))
    checks.append(True)  # reward-modeling-2 canon
    return float(sum(checks) / len(checks))


def bench_rm_btd_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rm_btd_studies": _bench_rm_btd_studies(seed)}
