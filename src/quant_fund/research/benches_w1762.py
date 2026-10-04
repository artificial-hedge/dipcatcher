"""Wave-1762 bench adapters: norse-myth-8 canon (SYNTHETIC only)."""

from quant_fund.models import (
    baldr_qa_studies,
    bragi_qa_studies,
    freya_qa_studies,
    heimdall_qa_studies,
    idunn_qa_studies,
    tyr_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17620


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_baldr_qa_studies_family(seed: int = _SEED + 0):
    """baldr_qa_studies: synthetic correctness bench."""
    return _finite_blob(baldr_qa_studies.bench_baldr_qa_studies(seed))


def bench_bragi_qa_studies_family(seed: int = _SEED + 1):
    """bragi_qa_studies: synthetic correctness bench."""
    return _finite_blob(bragi_qa_studies.bench_bragi_qa_studies(seed))


def bench_freya_qa_studies_family(seed: int = _SEED + 2):
    """freya_qa_studies: synthetic correctness bench."""
    return _finite_blob(freya_qa_studies.bench_freya_qa_studies(seed))


def bench_heimdall_qa_studies_family(seed: int = _SEED + 3):
    """heimdall_qa_studies: synthetic correctness bench."""
    return _finite_blob(heimdall_qa_studies.bench_heimdall_qa_studies(seed))


def bench_idunn_qa_studies_family(seed: int = _SEED + 4):
    """idunn_qa_studies: synthetic correctness bench."""
    return _finite_blob(idunn_qa_studies.bench_idunn_qa_studies(seed))


def bench_tyr_qa_studies_family(seed: int = _SEED + 5):
    """tyr_qa_studies: synthetic correctness bench."""
    return _finite_blob(tyr_qa_studies.bench_tyr_qa_studies(seed))
