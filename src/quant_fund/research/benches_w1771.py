"""Wave-1771 bench adapters: hittite-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    hannahanna_qa_studies,
    illuyanka_qa_studies,
    inara_qa_studies,
    kamrusepa_qa_studies,
    tarhunna_qa_studies,
    telepinus_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17710


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_hannahanna_qa_studies_family(seed: int = _SEED + 0):
    """hannahanna_qa_studies: synthetic correctness bench."""
    return _finite_blob(hannahanna_qa_studies.bench_hannahanna_qa_studies(seed))


def bench_illuyanka_qa_studies_family(seed: int = _SEED + 1):
    """illuyanka_qa_studies: synthetic correctness bench."""
    return _finite_blob(illuyanka_qa_studies.bench_illuyanka_qa_studies(seed))


def bench_inara_qa_studies_family(seed: int = _SEED + 2):
    """inara_qa_studies: synthetic correctness bench."""
    return _finite_blob(inara_qa_studies.bench_inara_qa_studies(seed))


def bench_kamrusepa_qa_studies_family(seed: int = _SEED + 3):
    """kamrusepa_qa_studies: synthetic correctness bench."""
    return _finite_blob(kamrusepa_qa_studies.bench_kamrusepa_qa_studies(seed))


def bench_tarhunna_qa_studies_family(seed: int = _SEED + 4):
    """tarhunna_qa_studies: synthetic correctness bench."""
    return _finite_blob(tarhunna_qa_studies.bench_tarhunna_qa_studies(seed))


def bench_telepinus_qa_studies_family(seed: int = _SEED + 5):
    """telepinus_qa_studies: synthetic correctness bench."""
    return _finite_blob(telepinus_qa_studies.bench_telepinus_qa_studies(seed))
