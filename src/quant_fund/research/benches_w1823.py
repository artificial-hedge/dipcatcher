"""Wave-1823 bench adapters: mongolian-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    almas2_qa_studies,
    khangai2_qa_studies,
    shunu2_qa_studies,
    sulde2_qa_studies,
    tengri2_qa_studies,
    ukerm2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18230


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_almas2_qa_studies_family(seed: int = _SEED + 0):
    """almas2_qa_studies: synthetic correctness bench."""
    return _finite_blob(almas2_qa_studies.bench_almas2_qa_studies(seed))


def bench_khangai2_qa_studies_family(seed: int = _SEED + 1):
    """khangai2_qa_studies: synthetic correctness bench."""
    return _finite_blob(khangai2_qa_studies.bench_khangai2_qa_studies(seed))


def bench_shunu2_qa_studies_family(seed: int = _SEED + 2):
    """shunu2_qa_studies: synthetic correctness bench."""
    return _finite_blob(shunu2_qa_studies.bench_shunu2_qa_studies(seed))


def bench_sulde2_qa_studies_family(seed: int = _SEED + 3):
    """sulde2_qa_studies: synthetic correctness bench."""
    return _finite_blob(sulde2_qa_studies.bench_sulde2_qa_studies(seed))


def bench_tengri2_qa_studies_family(seed: int = _SEED + 4):
    """tengri2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tengri2_qa_studies.bench_tengri2_qa_studies(seed))


def bench_ukerm2_qa_studies_family(seed: int = _SEED + 5):
    """ukerm2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ukerm2_qa_studies.bench_ukerm2_qa_studies(seed))
