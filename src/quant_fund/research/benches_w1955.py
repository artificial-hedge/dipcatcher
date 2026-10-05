"""Wave-1955 bench adapters: goetic-summons canon (SYNTHETIC only)."""

from quant_fund.models import (
    amdusias_qa_studies,
    andromalius_qa_studies,
    dantalion_qa_studies,
    decarabia_qa_studies,
    malphas_qa_studies,
    seere_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19550


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amdusias_qa_studies_family(seed: int = _SEED + 0):
    """amdusias_qa_studies: synthetic correctness bench."""
    return _finite_blob(amdusias_qa_studies.bench_amdusias_qa_studies(seed))


def bench_andromalius_qa_studies_family(seed: int = _SEED + 1):
    """andromalius_qa_studies: synthetic correctness bench."""
    return _finite_blob(andromalius_qa_studies.bench_andromalius_qa_studies(seed))


def bench_dantalion_qa_studies_family(seed: int = _SEED + 2):
    """dantalion_qa_studies: synthetic correctness bench."""
    return _finite_blob(dantalion_qa_studies.bench_dantalion_qa_studies(seed))


def bench_decarabia_qa_studies_family(seed: int = _SEED + 3):
    """decarabia_qa_studies: synthetic correctness bench."""
    return _finite_blob(decarabia_qa_studies.bench_decarabia_qa_studies(seed))


def bench_malphas_qa_studies_family(seed: int = _SEED + 4):
    """malphas_qa_studies: synthetic correctness bench."""
    return _finite_blob(malphas_qa_studies.bench_malphas_qa_studies(seed))


def bench_seere_qa_studies_family(seed: int = _SEED + 5):
    """seere_qa_studies: synthetic correctness bench."""
    return _finite_blob(seere_qa_studies.bench_seere_qa_studies(seed))
