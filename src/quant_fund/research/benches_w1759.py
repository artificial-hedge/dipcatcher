"""Wave-1759 bench adapters: aztec-deity-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    citlali_qa_studies,
    malinal_qa_studies,
    metzli_qa_studies,
    tepoz_qa_studies,
    tonaca_qa_studies,
    xochipilli_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17590


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_citlali_qa_studies_family(seed: int = _SEED + 0):
    """citlali_qa_studies: synthetic correctness bench."""
    return _finite_blob(citlali_qa_studies.bench_citlali_qa_studies(seed))


def bench_malinal_qa_studies_family(seed: int = _SEED + 1):
    """malinal_qa_studies: synthetic correctness bench."""
    return _finite_blob(malinal_qa_studies.bench_malinal_qa_studies(seed))


def bench_metzli_qa_studies_family(seed: int = _SEED + 2):
    """metzli_qa_studies: synthetic correctness bench."""
    return _finite_blob(metzli_qa_studies.bench_metzli_qa_studies(seed))


def bench_tepoz_qa_studies_family(seed: int = _SEED + 3):
    """tepoz_qa_studies: synthetic correctness bench."""
    return _finite_blob(tepoz_qa_studies.bench_tepoz_qa_studies(seed))


def bench_tonaca_qa_studies_family(seed: int = _SEED + 4):
    """tonaca_qa_studies: synthetic correctness bench."""
    return _finite_blob(tonaca_qa_studies.bench_tonaca_qa_studies(seed))


def bench_xochipilli_qa_studies_family(seed: int = _SEED + 5):
    """xochipilli_qa_studies: synthetic correctness bench."""
    return _finite_blob(xochipilli_qa_studies.bench_xochipilli_qa_studies(seed))
