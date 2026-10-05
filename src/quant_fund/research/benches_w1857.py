"""Wave-1857 bench adapters: romano-british-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    belatucadrus_qa_studies,
    cocidius_qa_studies,
    maponus_qa_studies,
    nemetona_qa_studies,
    rigisamus_qa_studies,
    sulis_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18570


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_belatucadrus_qa_studies_family(seed: int = _SEED + 0):
    """belatucadrus_qa_studies: synthetic correctness bench."""
    return _finite_blob(belatucadrus_qa_studies.bench_belatucadrus_qa_studies(seed))


def bench_cocidius_qa_studies_family(seed: int = _SEED + 1):
    """cocidius_qa_studies: synthetic correctness bench."""
    return _finite_blob(cocidius_qa_studies.bench_cocidius_qa_studies(seed))


def bench_maponus_qa_studies_family(seed: int = _SEED + 2):
    """maponus_qa_studies: synthetic correctness bench."""
    return _finite_blob(maponus_qa_studies.bench_maponus_qa_studies(seed))


def bench_nemetona_qa_studies_family(seed: int = _SEED + 3):
    """nemetona_qa_studies: synthetic correctness bench."""
    return _finite_blob(nemetona_qa_studies.bench_nemetona_qa_studies(seed))


def bench_rigisamus_qa_studies_family(seed: int = _SEED + 4):
    """rigisamus_qa_studies: synthetic correctness bench."""
    return _finite_blob(rigisamus_qa_studies.bench_rigisamus_qa_studies(seed))


def bench_sulis_qa_studies_family(seed: int = _SEED + 5):
    """sulis_qa_studies: synthetic correctness bench."""
    return _finite_blob(sulis_qa_studies.bench_sulis_qa_studies(seed))
