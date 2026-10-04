"""Wave-1889 bench adapters: eddic-lore-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    alvissmal_qa_studies,
    edda_lore_qa_studies,
    gylfaginning_prose_qa_studies,
    haddingjar_qa_studies,
    hyndluljod_qa_studies,
    rigsthula_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18890


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alvissmal_qa_studies_family(seed: int = _SEED + 0):
    """alvissmal_qa_studies: synthetic correctness bench."""
    return _finite_blob(alvissmal_qa_studies.bench_alvissmal_qa_studies(seed))


def bench_edda_lore_qa_studies_family(seed: int = _SEED + 1):
    """edda_lore_qa_studies: synthetic correctness bench."""
    return _finite_blob(edda_lore_qa_studies.bench_edda_lore_qa_studies(seed))


def bench_gylfaginning_prose_qa_studies_family(seed: int = _SEED + 2):
    """gylfaginning_prose_qa_studies: synthetic correctness bench."""
    return _finite_blob(gylfaginning_prose_qa_studies.bench_gylfaginning_prose_qa_studies(seed))


def bench_haddingjar_qa_studies_family(seed: int = _SEED + 3):
    """haddingjar_qa_studies: synthetic correctness bench."""
    return _finite_blob(haddingjar_qa_studies.bench_haddingjar_qa_studies(seed))


def bench_hyndluljod_qa_studies_family(seed: int = _SEED + 4):
    """hyndluljod_qa_studies: synthetic correctness bench."""
    return _finite_blob(hyndluljod_qa_studies.bench_hyndluljod_qa_studies(seed))


def bench_rigsthula_qa_studies_family(seed: int = _SEED + 5):
    """rigsthula_qa_studies: synthetic correctness bench."""
    return _finite_blob(rigsthula_qa_studies.bench_rigsthula_qa_studies(seed))
