"""Wave-1592 bench adapters: prosimian canon (SYNTHETIC only)."""

from quant_fund.models import (
    bushbaby_qa_studies,
    galago_qa_studies,
    indri_qa_studies,
    loris_qa_studies,
    potto_qa_studies,
    tarsier_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15920


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bushbaby_qa_studies_family(seed: int = _SEED + 0):
    """bushbaby_qa_studies: synthetic correctness bench."""
    return _finite_blob(bushbaby_qa_studies.bench_bushbaby_qa_studies(seed))


def bench_galago_qa_studies_family(seed: int = _SEED + 1):
    """galago_qa_studies: synthetic correctness bench."""
    return _finite_blob(galago_qa_studies.bench_galago_qa_studies(seed))


def bench_indri_qa_studies_family(seed: int = _SEED + 2):
    """indri_qa_studies: synthetic correctness bench."""
    return _finite_blob(indri_qa_studies.bench_indri_qa_studies(seed))


def bench_loris_qa_studies_family(seed: int = _SEED + 3):
    """loris_qa_studies: synthetic correctness bench."""
    return _finite_blob(loris_qa_studies.bench_loris_qa_studies(seed))


def bench_potto_qa_studies_family(seed: int = _SEED + 4):
    """potto_qa_studies: synthetic correctness bench."""
    return _finite_blob(potto_qa_studies.bench_potto_qa_studies(seed))


def bench_tarsier_qa_studies_family(seed: int = _SEED + 5):
    """tarsier_qa_studies: synthetic correctness bench."""
    return _finite_blob(tarsier_qa_studies.bench_tarsier_qa_studies(seed))
