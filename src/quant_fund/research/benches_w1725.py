"""Wave-1725 bench adapters: japanese-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    izanagi_qa_studies,
    izanami_qa_studies,
    kukunochi_qa_studies,
    omoikane_qa_studies,
    sarutahiko_qa_studies,
    uzume_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17250


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_izanagi_qa_studies_family(seed: int = _SEED + 0):
    """izanagi_qa_studies: synthetic correctness bench."""
    return _finite_blob(izanagi_qa_studies.bench_izanagi_qa_studies(seed))


def bench_izanami_qa_studies_family(seed: int = _SEED + 1):
    """izanami_qa_studies: synthetic correctness bench."""
    return _finite_blob(izanami_qa_studies.bench_izanami_qa_studies(seed))


def bench_kukunochi_qa_studies_family(seed: int = _SEED + 2):
    """kukunochi_qa_studies: synthetic correctness bench."""
    return _finite_blob(kukunochi_qa_studies.bench_kukunochi_qa_studies(seed))


def bench_omoikane_qa_studies_family(seed: int = _SEED + 3):
    """omoikane_qa_studies: synthetic correctness bench."""
    return _finite_blob(omoikane_qa_studies.bench_omoikane_qa_studies(seed))


def bench_sarutahiko_qa_studies_family(seed: int = _SEED + 4):
    """sarutahiko_qa_studies: synthetic correctness bench."""
    return _finite_blob(sarutahiko_qa_studies.bench_sarutahiko_qa_studies(seed))


def bench_uzume_qa_studies_family(seed: int = _SEED + 5):
    """uzume_qa_studies: synthetic correctness bench."""
    return _finite_blob(uzume_qa_studies.bench_uzume_qa_studies(seed))
