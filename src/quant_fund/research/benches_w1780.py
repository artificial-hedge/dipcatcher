"""Wave-1780 bench adapters: welsh-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    arawn_qa_studies,
    branwen_qa_studies,
    gwydion_qa_studies,
    lleu_qa_studies,
    lludd_qa_studies,
    taliesin_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17800


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_arawn_qa_studies_family(seed: int = _SEED + 0):
    """arawn_qa_studies: synthetic correctness bench."""
    return _finite_blob(arawn_qa_studies.bench_arawn_qa_studies(seed))


def bench_branwen_qa_studies_family(seed: int = _SEED + 1):
    """branwen_qa_studies: synthetic correctness bench."""
    return _finite_blob(branwen_qa_studies.bench_branwen_qa_studies(seed))


def bench_gwydion_qa_studies_family(seed: int = _SEED + 2):
    """gwydion_qa_studies: synthetic correctness bench."""
    return _finite_blob(gwydion_qa_studies.bench_gwydion_qa_studies(seed))


def bench_lleu_qa_studies_family(seed: int = _SEED + 3):
    """lleu_qa_studies: synthetic correctness bench."""
    return _finite_blob(lleu_qa_studies.bench_lleu_qa_studies(seed))


def bench_lludd_qa_studies_family(seed: int = _SEED + 4):
    """lludd_qa_studies: synthetic correctness bench."""
    return _finite_blob(lludd_qa_studies.bench_lludd_qa_studies(seed))


def bench_taliesin_qa_studies_family(seed: int = _SEED + 5):
    """taliesin_qa_studies: synthetic correctness bench."""
    return _finite_blob(taliesin_qa_studies.bench_taliesin_qa_studies(seed))
