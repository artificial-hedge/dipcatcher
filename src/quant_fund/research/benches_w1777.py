"""Wave-1777 bench adapters: indonesian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    barong_qa_studies,
    garuda_qa_studies,
    nyai_qa_studies,
    raksasa_qa_studies,
    rangda_qa_studies,
    semar_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17770


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_barong_qa_studies_family(seed: int = _SEED + 0):
    """barong_qa_studies: synthetic correctness bench."""
    return _finite_blob(barong_qa_studies.bench_barong_qa_studies(seed))


def bench_garuda_qa_studies_family(seed: int = _SEED + 1):
    """garuda_qa_studies: synthetic correctness bench."""
    return _finite_blob(garuda_qa_studies.bench_garuda_qa_studies(seed))


def bench_nyai_qa_studies_family(seed: int = _SEED + 2):
    """nyai_qa_studies: synthetic correctness bench."""
    return _finite_blob(nyai_qa_studies.bench_nyai_qa_studies(seed))


def bench_raksasa_qa_studies_family(seed: int = _SEED + 3):
    """raksasa_qa_studies: synthetic correctness bench."""
    return _finite_blob(raksasa_qa_studies.bench_raksasa_qa_studies(seed))


def bench_rangda_qa_studies_family(seed: int = _SEED + 4):
    """rangda_qa_studies: synthetic correctness bench."""
    return _finite_blob(rangda_qa_studies.bench_rangda_qa_studies(seed))


def bench_semar_qa_studies_family(seed: int = _SEED + 5):
    """semar_qa_studies: synthetic correctness bench."""
    return _finite_blob(semar_qa_studies.bench_semar_qa_studies(seed))
