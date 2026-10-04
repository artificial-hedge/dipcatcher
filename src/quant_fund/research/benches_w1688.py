"""Wave-1688 bench adapters: roman-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    indiges_qa_studies,
    lar_qa_studies,
    numen_qa_studies,
    penates_qa_studies,
    terminus_qa_studies,
    vertumnus_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16880


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_indiges_qa_studies_family(seed: int = _SEED + 0):
    """indiges_qa_studies: synthetic correctness bench."""
    return _finite_blob(indiges_qa_studies.bench_indiges_qa_studies(seed))


def bench_lar_qa_studies_family(seed: int = _SEED + 1):
    """lar_qa_studies: synthetic correctness bench."""
    return _finite_blob(lar_qa_studies.bench_lar_qa_studies(seed))


def bench_numen_qa_studies_family(seed: int = _SEED + 2):
    """numen_qa_studies: synthetic correctness bench."""
    return _finite_blob(numen_qa_studies.bench_numen_qa_studies(seed))


def bench_penates_qa_studies_family(seed: int = _SEED + 3):
    """penates_qa_studies: synthetic correctness bench."""
    return _finite_blob(penates_qa_studies.bench_penates_qa_studies(seed))


def bench_terminus_qa_studies_family(seed: int = _SEED + 4):
    """terminus_qa_studies: synthetic correctness bench."""
    return _finite_blob(terminus_qa_studies.bench_terminus_qa_studies(seed))


def bench_vertumnus_qa_studies_family(seed: int = _SEED + 5):
    """vertumnus_qa_studies: synthetic correctness bench."""
    return _finite_blob(vertumnus_qa_studies.bench_vertumnus_qa_studies(seed))
