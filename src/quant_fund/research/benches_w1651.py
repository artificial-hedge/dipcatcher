"""Wave-1651 bench adapters: australian-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    awgy_qa_studies,
    kuritja_qa_studies,
    minka_qa_studies,
    papin_qa_studies,
    yara_qa_studies,
    yowie_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16510


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_awgy_qa_studies_family(seed: int = _SEED + 0):
    """awgy_qa_studies: synthetic correctness bench."""
    return _finite_blob(awgy_qa_studies.bench_awgy_qa_studies(seed))


def bench_kuritja_qa_studies_family(seed: int = _SEED + 1):
    """kuritja_qa_studies: synthetic correctness bench."""
    return _finite_blob(kuritja_qa_studies.bench_kuritja_qa_studies(seed))


def bench_minka_qa_studies_family(seed: int = _SEED + 2):
    """minka_qa_studies: synthetic correctness bench."""
    return _finite_blob(minka_qa_studies.bench_minka_qa_studies(seed))


def bench_papin_qa_studies_family(seed: int = _SEED + 3):
    """papin_qa_studies: synthetic correctness bench."""
    return _finite_blob(papin_qa_studies.bench_papin_qa_studies(seed))


def bench_yara_qa_studies_family(seed: int = _SEED + 4):
    """yara_qa_studies: synthetic correctness bench."""
    return _finite_blob(yara_qa_studies.bench_yara_qa_studies(seed))


def bench_yowie_qa_studies_family(seed: int = _SEED + 5):
    """yowie_qa_studies: synthetic correctness bench."""
    return _finite_blob(yowie_qa_studies.bench_yowie_qa_studies(seed))
