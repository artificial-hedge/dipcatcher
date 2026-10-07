"""Wave-1693 bench adapters: mesopotamian-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    asag_qa_studies,
    edimmu_qa_studies,
    galla_qa_studies,
    lamassu_qa_studies,
    shedu_qa_studies,
    utukku_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16930


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_asag_qa_studies_family(seed: int = _SEED + 0):
    """asag_qa_studies: synthetic correctness bench."""
    return _finite_blob(asag_qa_studies.bench_asag_qa_studies(seed))


def bench_edimmu_qa_studies_family(seed: int = _SEED + 1):
    """edimmu_qa_studies: synthetic correctness bench."""
    return _finite_blob(edimmu_qa_studies.bench_edimmu_qa_studies(seed))


def bench_galla_qa_studies_family(seed: int = _SEED + 2):
    """galla_qa_studies: synthetic correctness bench."""
    return _finite_blob(galla_qa_studies.bench_galla_qa_studies(seed))


def bench_lamassu_qa_studies_family(seed: int = _SEED + 3):
    """lamassu_qa_studies: synthetic correctness bench."""
    return _finite_blob(lamassu_qa_studies.bench_lamassu_qa_studies(seed))


def bench_shedu_qa_studies_family(seed: int = _SEED + 4):
    """shedu_qa_studies: synthetic correctness bench."""
    return _finite_blob(shedu_qa_studies.bench_shedu_qa_studies(seed))


def bench_utukku_qa_studies_family(seed: int = _SEED + 5):
    """utukku_qa_studies: synthetic correctness bench."""
    return _finite_blob(utukku_qa_studies.bench_utukku_qa_studies(seed))
