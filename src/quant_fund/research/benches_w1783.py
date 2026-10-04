"""Wave-1783 bench adapters: norse-myth-10 canon (SYNTHETIC only)."""

from quant_fund.models import (
    baldur_qa_studies,
    freyr_qa_studies,
    hermodr_qa_studies,
    hodr_qa_studies,
    njord_qa_studies,
    skadi_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17830


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_baldur_qa_studies_family(seed: int = _SEED + 0):
    """baldur_qa_studies: synthetic correctness bench."""
    return _finite_blob(baldur_qa_studies.bench_baldur_qa_studies(seed))


def bench_freyr_qa_studies_family(seed: int = _SEED + 1):
    """freyr_qa_studies: synthetic correctness bench."""
    return _finite_blob(freyr_qa_studies.bench_freyr_qa_studies(seed))


def bench_hermodr_qa_studies_family(seed: int = _SEED + 2):
    """hermodr_qa_studies: synthetic correctness bench."""
    return _finite_blob(hermodr_qa_studies.bench_hermodr_qa_studies(seed))


def bench_hodr_qa_studies_family(seed: int = _SEED + 3):
    """hodr_qa_studies: synthetic correctness bench."""
    return _finite_blob(hodr_qa_studies.bench_hodr_qa_studies(seed))


def bench_njord_qa_studies_family(seed: int = _SEED + 4):
    """njord_qa_studies: synthetic correctness bench."""
    return _finite_blob(njord_qa_studies.bench_njord_qa_studies(seed))


def bench_skadi_qa_studies_family(seed: int = _SEED + 5):
    """skadi_qa_studies: synthetic correctness bench."""
    return _finite_blob(skadi_qa_studies.bench_skadi_qa_studies(seed))
