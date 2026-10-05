"""Wave-1883 bench adapters: saharan canon (SYNTHETIC only)."""

from quant_fund.models import (
    aewan_qa_studies,
    banguilet_qa_studies,
    hemmi_qa_studies,
    maziun_qa_studies,
    tissardal_qa_studies,
    zilalsen_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18830


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aewan_qa_studies_family(seed: int = _SEED + 0):
    """aewan_qa_studies: synthetic correctness bench."""
    return _finite_blob(aewan_qa_studies.bench_aewan_qa_studies(seed))


def bench_banguilet_qa_studies_family(seed: int = _SEED + 1):
    """banguilet_qa_studies: synthetic correctness bench."""
    return _finite_blob(banguilet_qa_studies.bench_banguilet_qa_studies(seed))


def bench_hemmi_qa_studies_family(seed: int = _SEED + 2):
    """hemmi_qa_studies: synthetic correctness bench."""
    return _finite_blob(hemmi_qa_studies.bench_hemmi_qa_studies(seed))


def bench_maziun_qa_studies_family(seed: int = _SEED + 3):
    """maziun_qa_studies: synthetic correctness bench."""
    return _finite_blob(maziun_qa_studies.bench_maziun_qa_studies(seed))


def bench_tissardal_qa_studies_family(seed: int = _SEED + 4):
    """tissardal_qa_studies: synthetic correctness bench."""
    return _finite_blob(tissardal_qa_studies.bench_tissardal_qa_studies(seed))


def bench_zilalsen_qa_studies_family(seed: int = _SEED + 5):
    """zilalsen_qa_studies: synthetic correctness bench."""
    return _finite_blob(zilalsen_qa_studies.bench_zilalsen_qa_studies(seed))
