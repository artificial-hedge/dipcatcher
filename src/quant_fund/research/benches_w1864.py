"""Wave-1864 bench adapters: arthurian-5 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bors_qa_studies,
    culhwch_qa_studies,
    dinadan_qa_studies,
    palamedes_qa_studies,
    safir_qa_studies,
    segwarides_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18640


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bors_qa_studies_family(seed: int = _SEED + 0):
    """bors_qa_studies: synthetic correctness bench."""
    return _finite_blob(bors_qa_studies.bench_bors_qa_studies(seed))


def bench_culhwch_qa_studies_family(seed: int = _SEED + 1):
    """culhwch_qa_studies: synthetic correctness bench."""
    return _finite_blob(culhwch_qa_studies.bench_culhwch_qa_studies(seed))


def bench_dinadan_qa_studies_family(seed: int = _SEED + 2):
    """dinadan_qa_studies: synthetic correctness bench."""
    return _finite_blob(dinadan_qa_studies.bench_dinadan_qa_studies(seed))


def bench_palamedes_qa_studies_family(seed: int = _SEED + 3):
    """palamedes_qa_studies: synthetic correctness bench."""
    return _finite_blob(palamedes_qa_studies.bench_palamedes_qa_studies(seed))


def bench_safir_qa_studies_family(seed: int = _SEED + 4):
    """safir_qa_studies: synthetic correctness bench."""
    return _finite_blob(safir_qa_studies.bench_safir_qa_studies(seed))


def bench_segwarides_qa_studies_family(seed: int = _SEED + 5):
    """segwarides_qa_studies: synthetic correctness bench."""
    return _finite_blob(segwarides_qa_studies.bench_segwarides_qa_studies(seed))
