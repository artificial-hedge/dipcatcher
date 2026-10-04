"""Wave-1483 bench adapters: mammal canon (SYNTHETIC only)."""

from quant_fund.models import (
    coyote_qa_studies,
    ferret_qa_studies,
    jackal_qa_studies,
    marmot_qa_studies,
    moose_qa_studies,
    raccoon_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14830


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_coyote_qa_studies_family(seed: int = _SEED + 0):
    """coyote_qa_studies: synthetic correctness bench."""
    return _finite_blob(coyote_qa_studies.bench_coyote_qa_studies(seed))


def bench_ferret_qa_studies_family(seed: int = _SEED + 1):
    """ferret_qa_studies: synthetic correctness bench."""
    return _finite_blob(ferret_qa_studies.bench_ferret_qa_studies(seed))


def bench_jackal_qa_studies_family(seed: int = _SEED + 2):
    """jackal_qa_studies: synthetic correctness bench."""
    return _finite_blob(jackal_qa_studies.bench_jackal_qa_studies(seed))


def bench_marmot_qa_studies_family(seed: int = _SEED + 3):
    """marmot_qa_studies: synthetic correctness bench."""
    return _finite_blob(marmot_qa_studies.bench_marmot_qa_studies(seed))


def bench_moose_qa_studies_family(seed: int = _SEED + 4):
    """moose_qa_studies: synthetic correctness bench."""
    return _finite_blob(moose_qa_studies.bench_moose_qa_studies(seed))


def bench_raccoon_qa_studies_family(seed: int = _SEED + 5):
    """raccoon_qa_studies: synthetic correctness bench."""
    return _finite_blob(raccoon_qa_studies.bench_raccoon_qa_studies(seed))
