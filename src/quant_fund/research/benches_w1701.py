"""Wave-1701 bench adapters: incan-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    illapa_qa_studies,
    inti_qa_studies,
    mamaquilla_qa_studies,
    pachamama_qa_studies,
    supay_qa_studies,
    viracocha_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17010


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


def bench_illapa_qa_studies_family(seed: int = _SEED + 0):
    """illapa_qa_studies: synthetic correctness bench."""
    return _finite_blob(illapa_qa_studies.bench_illapa_qa_studies(seed))


def bench_inti_qa_studies_family(seed: int = _SEED + 1):
    """inti_qa_studies: synthetic correctness bench."""
    return _finite_blob(inti_qa_studies.bench_inti_qa_studies(seed))


def bench_mamaquilla_qa_studies_family(seed: int = _SEED + 2):
    """mamaquilla_qa_studies: synthetic correctness bench."""
    return _finite_blob(mamaquilla_qa_studies.bench_mamaquilla_qa_studies(seed))


def bench_pachamama_qa_studies_family(seed: int = _SEED + 3):
    """pachamama_qa_studies: synthetic correctness bench."""
    return _finite_blob(pachamama_qa_studies.bench_pachamama_qa_studies(seed))


def bench_supay_qa_studies_family(seed: int = _SEED + 4):
    """supay_qa_studies: synthetic correctness bench."""
    return _finite_blob(supay_qa_studies.bench_supay_qa_studies(seed))


def bench_viracocha_qa_studies_family(seed: int = _SEED + 5):
    """viracocha_qa_studies: synthetic correctness bench."""
    return _finite_blob(viracocha_qa_studies.bench_viracocha_qa_studies(seed))
