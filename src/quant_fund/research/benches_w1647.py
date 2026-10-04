"""Wave-1647 bench adapters: egyptian-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    akhekh_qa_studies,
    ammit_qa_studies,
    apophis_qa_studies,
    bes_qa_studies,
    sekhmet_qa_studies,
    sphairo_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16470


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_akhekh_qa_studies_family(seed: int = _SEED + 0):
    """akhekh_qa_studies: synthetic correctness bench."""
    return _finite_blob(akhekh_qa_studies.bench_akhekh_qa_studies(seed))


def bench_ammit_qa_studies_family(seed: int = _SEED + 1):
    """ammit_qa_studies: synthetic correctness bench."""
    return _finite_blob(ammit_qa_studies.bench_ammit_qa_studies(seed))


def bench_apophis_qa_studies_family(seed: int = _SEED + 2):
    """apophis_qa_studies: synthetic correctness bench."""
    return _finite_blob(apophis_qa_studies.bench_apophis_qa_studies(seed))


def bench_bes_qa_studies_family(seed: int = _SEED + 3):
    """bes_qa_studies: synthetic correctness bench."""
    return _finite_blob(bes_qa_studies.bench_bes_qa_studies(seed))


def bench_sekhmet_qa_studies_family(seed: int = _SEED + 4):
    """sekhmet_qa_studies: synthetic correctness bench."""
    return _finite_blob(sekhmet_qa_studies.bench_sekhmet_qa_studies(seed))


def bench_sphairo_qa_studies_family(seed: int = _SEED + 5):
    """sphairo_qa_studies: synthetic correctness bench."""
    return _finite_blob(sphairo_qa_studies.bench_sphairo_qa_studies(seed))
