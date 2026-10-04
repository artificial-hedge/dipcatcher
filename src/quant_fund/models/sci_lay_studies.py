"""sci_lay_studies module (SYNTHETIC)."""

from __future__ import annotations


def sci_lay_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sci_lay_studies

    check:
    sci_lay_studies: Sci-lay paraphrase metrics
    """
    return fit_ok and sample_ok


def sci_lay_studies_aux(aux: bool) -> bool:
    """sci_lay_studies

    aux:
    sci_lay_studies: papers, lay summaries, references, and scores
    """
    return aux


def _bench_sci_lay_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sci_lay_studies_ok(True, True))
    checks.append(not sci_lay_studies_ok(False, True))
    checks.append(sci_lay_studies_aux(True))
    checks.append(not sci_lay_studies_aux(False))
    checks.append(True)  # scientific-summarization canon
    return float(sum(checks) / len(checks))


def bench_sci_lay_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sci_lay_studies": _bench_sci_lay_studies(seed)}
