"""Wave-1671 bench adapters: scandinavian-folk canon (SYNTHETIC only)."""

from quant_fund.models import (
    drakk_qa_studies,
    grimr_qa_studies,
    hildr_qa_studies,
    mare_qa_studies,
    nisse_qa_studies,
    sigrun_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16710


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


def bench_drakk_qa_studies_family(seed: int = _SEED + 0):
    """drakk_qa_studies: synthetic correctness bench."""
    return _finite_blob(drakk_qa_studies.bench_drakk_qa_studies(seed))


def bench_grimr_qa_studies_family(seed: int = _SEED + 1):
    """grimr_qa_studies: synthetic correctness bench."""
    return _finite_blob(grimr_qa_studies.bench_grimr_qa_studies(seed))


def bench_hildr_qa_studies_family(seed: int = _SEED + 2):
    """hildr_qa_studies: synthetic correctness bench."""
    return _finite_blob(hildr_qa_studies.bench_hildr_qa_studies(seed))


def bench_mare_qa_studies_family(seed: int = _SEED + 3):
    """mare_qa_studies: synthetic correctness bench."""
    return _finite_blob(mare_qa_studies.bench_mare_qa_studies(seed))


def bench_nisse_qa_studies_family(seed: int = _SEED + 4):
    """nisse_qa_studies: synthetic correctness bench."""
    return _finite_blob(nisse_qa_studies.bench_nisse_qa_studies(seed))


def bench_sigrun_qa_studies_family(seed: int = _SEED + 5):
    """sigrun_qa_studies: synthetic correctness bench."""
    return _finite_blob(sigrun_qa_studies.bench_sigrun_qa_studies(seed))
