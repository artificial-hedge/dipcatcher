"""Wave-1673 bench adapters: hindu-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    apsara_qa_studies,
    asura_qa_studies,
    gandharva_qa_studies,
    naga_qa_studies,
    rakshasa_qa_studies,
    yaksha_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16730


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


def bench_apsara_qa_studies_family(seed: int = _SEED + 0):
    """apsara_qa_studies: synthetic correctness bench."""
    return _finite_blob(apsara_qa_studies.bench_apsara_qa_studies(seed))


def bench_asura_qa_studies_family(seed: int = _SEED + 1):
    """asura_qa_studies: synthetic correctness bench."""
    return _finite_blob(asura_qa_studies.bench_asura_qa_studies(seed))


def bench_gandharva_qa_studies_family(seed: int = _SEED + 2):
    """gandharva_qa_studies: synthetic correctness bench."""
    return _finite_blob(gandharva_qa_studies.bench_gandharva_qa_studies(seed))


def bench_naga_qa_studies_family(seed: int = _SEED + 3):
    """naga_qa_studies: synthetic correctness bench."""
    return _finite_blob(naga_qa_studies.bench_naga_qa_studies(seed))


def bench_rakshasa_qa_studies_family(seed: int = _SEED + 4):
    """rakshasa_qa_studies: synthetic correctness bench."""
    return _finite_blob(rakshasa_qa_studies.bench_rakshasa_qa_studies(seed))


def bench_yaksha_qa_studies_family(seed: int = _SEED + 5):
    """yaksha_qa_studies: synthetic correctness bench."""
    return _finite_blob(yaksha_qa_studies.bench_yaksha_qa_studies(seed))
