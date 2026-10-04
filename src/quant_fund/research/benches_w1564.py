"""Wave-1564 bench adapters: cephalopod canon (SYNTHETIC only)."""

from quant_fund.models import (
    bobtail_squid_qa_studies,
    cuttlefish_qa_studies,
    nautilus_qa_studies,
    nudibranch_qa_studies,
    sea_slug_qa_studies,
    vampire_squid_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15640


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bobtail_squid_qa_studies_family(seed: int = _SEED + 0):
    """bobtail_squid_qa_studies: synthetic correctness bench."""
    return _finite_blob(bobtail_squid_qa_studies.bench_bobtail_squid_qa_studies(seed))


def bench_cuttlefish_qa_studies_family(seed: int = _SEED + 1):
    """cuttlefish_qa_studies: synthetic correctness bench."""
    return _finite_blob(cuttlefish_qa_studies.bench_cuttlefish_qa_studies(seed))


def bench_nautilus_qa_studies_family(seed: int = _SEED + 2):
    """nautilus_qa_studies: synthetic correctness bench."""
    return _finite_blob(nautilus_qa_studies.bench_nautilus_qa_studies(seed))


def bench_nudibranch_qa_studies_family(seed: int = _SEED + 3):
    """nudibranch_qa_studies: synthetic correctness bench."""
    return _finite_blob(nudibranch_qa_studies.bench_nudibranch_qa_studies(seed))


def bench_sea_slug_qa_studies_family(seed: int = _SEED + 4):
    """sea_slug_qa_studies: synthetic correctness bench."""
    return _finite_blob(sea_slug_qa_studies.bench_sea_slug_qa_studies(seed))


def bench_vampire_squid_qa_studies_family(seed: int = _SEED + 5):
    """vampire_squid_qa_studies: synthetic correctness bench."""
    return _finite_blob(vampire_squid_qa_studies.bench_vampire_squid_qa_studies(seed))
