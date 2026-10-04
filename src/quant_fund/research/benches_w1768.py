"""Wave-1768 bench adapters: african-myth-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    buluku_qa_studies,
    chukwu_qa_studies,
    eshu_qa_studies,
    mawu_qa_studies,
    nyambi_qa_studies,
    oshumare_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17680


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_buluku_qa_studies_family(seed: int = _SEED + 0):
    """buluku_qa_studies: synthetic correctness bench."""
    return _finite_blob(buluku_qa_studies.bench_buluku_qa_studies(seed))


def bench_chukwu_qa_studies_family(seed: int = _SEED + 1):
    """chukwu_qa_studies: synthetic correctness bench."""
    return _finite_blob(chukwu_qa_studies.bench_chukwu_qa_studies(seed))


def bench_eshu_qa_studies_family(seed: int = _SEED + 2):
    """eshu_qa_studies: synthetic correctness bench."""
    return _finite_blob(eshu_qa_studies.bench_eshu_qa_studies(seed))


def bench_mawu_qa_studies_family(seed: int = _SEED + 3):
    """mawu_qa_studies: synthetic correctness bench."""
    return _finite_blob(mawu_qa_studies.bench_mawu_qa_studies(seed))


def bench_nyambi_qa_studies_family(seed: int = _SEED + 4):
    """nyambi_qa_studies: synthetic correctness bench."""
    return _finite_blob(nyambi_qa_studies.bench_nyambi_qa_studies(seed))


def bench_oshumare_qa_studies_family(seed: int = _SEED + 5):
    """oshumare_qa_studies: synthetic correctness bench."""
    return _finite_blob(oshumare_qa_studies.bench_oshumare_qa_studies(seed))
