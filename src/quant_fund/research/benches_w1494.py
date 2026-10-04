"""Wave-1494 bench adapters: reptile-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    anole_qa_studies,
    chameleon_qa_studies,
    hognose_qa_studies,
    skink_qa_studies,
    terrapin_qa_studies,
    tuatara_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14940


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_anole_qa_studies_family(seed: int = _SEED + 0):
    """anole_qa_studies: synthetic correctness bench."""
    return _finite_blob(anole_qa_studies.bench_anole_qa_studies(seed))


def bench_chameleon_qa_studies_family(seed: int = _SEED + 1):
    """chameleon_qa_studies: synthetic correctness bench."""
    return _finite_blob(chameleon_qa_studies.bench_chameleon_qa_studies(seed))


def bench_hognose_qa_studies_family(seed: int = _SEED + 2):
    """hognose_qa_studies: synthetic correctness bench."""
    return _finite_blob(hognose_qa_studies.bench_hognose_qa_studies(seed))


def bench_skink_qa_studies_family(seed: int = _SEED + 3):
    """skink_qa_studies: synthetic correctness bench."""
    return _finite_blob(skink_qa_studies.bench_skink_qa_studies(seed))


def bench_terrapin_qa_studies_family(seed: int = _SEED + 4):
    """terrapin_qa_studies: synthetic correctness bench."""
    return _finite_blob(terrapin_qa_studies.bench_terrapin_qa_studies(seed))


def bench_tuatara_qa_studies_family(seed: int = _SEED + 5):
    """tuatara_qa_studies: synthetic correctness bench."""
    return _finite_blob(tuatara_qa_studies.bench_tuatara_qa_studies(seed))
