"""Wave-1709 bench adapters: hittite-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    arinniti_qa_studies,
    hannahanna_qa_studies,
    inara_qa_studies,
    kamrusepa_qa_studies,
    tarhunna_qa_studies,
    telepinu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17090


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


def bench_arinniti_qa_studies_family(seed: int = _SEED + 0):
    """arinniti_qa_studies: synthetic correctness bench."""
    return _finite_blob(arinniti_qa_studies.bench_arinniti_qa_studies(seed))


def bench_hannahanna_qa_studies_family(seed: int = _SEED + 1):
    """hannahanna_qa_studies: synthetic correctness bench."""
    return _finite_blob(hannahanna_qa_studies.bench_hannahanna_qa_studies(seed))


def bench_inara_qa_studies_family(seed: int = _SEED + 2):
    """inara_qa_studies: synthetic correctness bench."""
    return _finite_blob(inara_qa_studies.bench_inara_qa_studies(seed))


def bench_kamrusepa_qa_studies_family(seed: int = _SEED + 3):
    """kamrusepa_qa_studies: synthetic correctness bench."""
    return _finite_blob(kamrusepa_qa_studies.bench_kamrusepa_qa_studies(seed))


def bench_tarhunna_qa_studies_family(seed: int = _SEED + 4):
    """tarhunna_qa_studies: synthetic correctness bench."""
    return _finite_blob(tarhunna_qa_studies.bench_tarhunna_qa_studies(seed))


def bench_telepinu_qa_studies_family(seed: int = _SEED + 5):
    """telepinu_qa_studies: synthetic correctness bench."""
    return _finite_blob(telepinu_qa_studies.bench_telepinu_qa_studies(seed))
