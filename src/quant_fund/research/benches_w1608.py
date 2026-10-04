"""Wave-1608 bench adapters: savanna-herd canon (SYNTHETIC only)."""

from quant_fund.models import (
    buffalo_qa_studies,
    kob_qa_studies,
    lechwe_qa_studies,
    rhino_qa_studies,
    roan_qa_studies,
    warthog_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16080


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_buffalo_qa_studies_family(seed: int = _SEED + 0):
    """buffalo_qa_studies: synthetic correctness bench."""
    return _finite_blob(buffalo_qa_studies.bench_buffalo_qa_studies(seed))


def bench_kob_qa_studies_family(seed: int = _SEED + 1):
    """kob_qa_studies: synthetic correctness bench."""
    return _finite_blob(kob_qa_studies.bench_kob_qa_studies(seed))


def bench_lechwe_qa_studies_family(seed: int = _SEED + 2):
    """lechwe_qa_studies: synthetic correctness bench."""
    return _finite_blob(lechwe_qa_studies.bench_lechwe_qa_studies(seed))


def bench_rhino_qa_studies_family(seed: int = _SEED + 3):
    """rhino_qa_studies: synthetic correctness bench."""
    return _finite_blob(rhino_qa_studies.bench_rhino_qa_studies(seed))


def bench_roan_qa_studies_family(seed: int = _SEED + 4):
    """roan_qa_studies: synthetic correctness bench."""
    return _finite_blob(roan_qa_studies.bench_roan_qa_studies(seed))


def bench_warthog_qa_studies_family(seed: int = _SEED + 5):
    """warthog_qa_studies: synthetic correctness bench."""
    return _finite_blob(warthog_qa_studies.bench_warthog_qa_studies(seed))
