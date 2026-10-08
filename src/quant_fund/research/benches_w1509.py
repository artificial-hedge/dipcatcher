"""Wave-1509 bench adapters: songbird-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bunting_qa_studies,
    grosbeak_qa_studies,
    nuthatch_qa_studies,
    tanager_qa_studies,
    titmouse_qa_studies,
    vireo_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15090


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


def bench_bunting_qa_studies_family(seed: int = _SEED + 0):
    """bunting_qa_studies: synthetic correctness bench."""
    return _finite_blob(bunting_qa_studies.bench_bunting_qa_studies(seed))


def bench_grosbeak_qa_studies_family(seed: int = _SEED + 1):
    """grosbeak_qa_studies: synthetic correctness bench."""
    return _finite_blob(grosbeak_qa_studies.bench_grosbeak_qa_studies(seed))


def bench_nuthatch_qa_studies_family(seed: int = _SEED + 2):
    """nuthatch_qa_studies: synthetic correctness bench."""
    return _finite_blob(nuthatch_qa_studies.bench_nuthatch_qa_studies(seed))


def bench_tanager_qa_studies_family(seed: int = _SEED + 3):
    """tanager_qa_studies: synthetic correctness bench."""
    return _finite_blob(tanager_qa_studies.bench_tanager_qa_studies(seed))


def bench_titmouse_qa_studies_family(seed: int = _SEED + 4):
    """titmouse_qa_studies: synthetic correctness bench."""
    return _finite_blob(titmouse_qa_studies.bench_titmouse_qa_studies(seed))


def bench_vireo_qa_studies_family(seed: int = _SEED + 5):
    """vireo_qa_studies: synthetic correctness bench."""
    return _finite_blob(vireo_qa_studies.bench_vireo_qa_studies(seed))
