"""Wave-1691 bench adapters: mesopotamian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    abzu_qa_studies,
    enki_qa_studies,
    enlil_qa_studies,
    nanna_qa_studies,
    tiamat_qa_studies,
    utu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16910


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


def bench_abzu_qa_studies_family(seed: int = _SEED + 0):
    """abzu_qa_studies: synthetic correctness bench."""
    return _finite_blob(abzu_qa_studies.bench_abzu_qa_studies(seed))


def bench_enki_qa_studies_family(seed: int = _SEED + 1):
    """enki_qa_studies: synthetic correctness bench."""
    return _finite_blob(enki_qa_studies.bench_enki_qa_studies(seed))


def bench_enlil_qa_studies_family(seed: int = _SEED + 2):
    """enlil_qa_studies: synthetic correctness bench."""
    return _finite_blob(enlil_qa_studies.bench_enlil_qa_studies(seed))


def bench_nanna_qa_studies_family(seed: int = _SEED + 3):
    """nanna_qa_studies: synthetic correctness bench."""
    return _finite_blob(nanna_qa_studies.bench_nanna_qa_studies(seed))


def bench_tiamat_qa_studies_family(seed: int = _SEED + 4):
    """tiamat_qa_studies: synthetic correctness bench."""
    return _finite_blob(tiamat_qa_studies.bench_tiamat_qa_studies(seed))


def bench_utu_qa_studies_family(seed: int = _SEED + 5):
    """utu_qa_studies: synthetic correctness bench."""
    return _finite_blob(utu_qa_studies.bench_utu_qa_studies(seed))
