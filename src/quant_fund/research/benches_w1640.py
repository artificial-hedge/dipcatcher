"""Wave-1640 bench adapters: guardian-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    byakko_qa_studies,
    genbu_qa_studies,
    kirin_2_qa_studies,
    kohryu_qa_studies,
    seiryu_qa_studies,
    suzaku_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16400


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


def bench_byakko_qa_studies_family(seed: int = _SEED + 0):
    """byakko_qa_studies: synthetic correctness bench."""
    return _finite_blob(byakko_qa_studies.bench_byakko_qa_studies(seed))


def bench_genbu_qa_studies_family(seed: int = _SEED + 1):
    """genbu_qa_studies: synthetic correctness bench."""
    return _finite_blob(genbu_qa_studies.bench_genbu_qa_studies(seed))


def bench_kirin_2_qa_studies_family(seed: int = _SEED + 2):
    """kirin_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kirin_2_qa_studies.bench_kirin_2_qa_studies(seed))


def bench_kohryu_qa_studies_family(seed: int = _SEED + 3):
    """kohryu_qa_studies: synthetic correctness bench."""
    return _finite_blob(kohryu_qa_studies.bench_kohryu_qa_studies(seed))


def bench_seiryu_qa_studies_family(seed: int = _SEED + 4):
    """seiryu_qa_studies: synthetic correctness bench."""
    return _finite_blob(seiryu_qa_studies.bench_seiryu_qa_studies(seed))


def bench_suzaku_qa_studies_family(seed: int = _SEED + 5):
    """suzaku_qa_studies: synthetic correctness bench."""
    return _finite_blob(suzaku_qa_studies.bench_suzaku_qa_studies(seed))
