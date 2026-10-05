"""Wave-1794 bench adapters: maori-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    maru2_qa_studies,
    pere_qa_studies,
    rongomai_qa_studies,
    tuhi2_qa_studies,
    uenuku2_qa_studies,
    wairere_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17940


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_maru2_qa_studies_family(seed: int = _SEED + 0):
    """maru2_qa_studies: synthetic correctness bench."""
    return _finite_blob(maru2_qa_studies.bench_maru2_qa_studies(seed))


def bench_pere_qa_studies_family(seed: int = _SEED + 1):
    """pere_qa_studies: synthetic correctness bench."""
    return _finite_blob(pere_qa_studies.bench_pere_qa_studies(seed))


def bench_rongomai_qa_studies_family(seed: int = _SEED + 2):
    """rongomai_qa_studies: synthetic correctness bench."""
    return _finite_blob(rongomai_qa_studies.bench_rongomai_qa_studies(seed))


def bench_tuhi2_qa_studies_family(seed: int = _SEED + 3):
    """tuhi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tuhi2_qa_studies.bench_tuhi2_qa_studies(seed))


def bench_uenuku2_qa_studies_family(seed: int = _SEED + 4):
    """uenuku2_qa_studies: synthetic correctness bench."""
    return _finite_blob(uenuku2_qa_studies.bench_uenuku2_qa_studies(seed))


def bench_wairere_qa_studies_family(seed: int = _SEED + 5):
    """wairere_qa_studies: synthetic correctness bench."""
    return _finite_blob(wairere_qa_studies.bench_wairere_qa_studies(seed))
