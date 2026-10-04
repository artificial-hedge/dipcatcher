"""Wave-1782 bench adapters: egyptian-5 canon (SYNTHETIC only)."""

from quant_fund.models import (
    geb_qa_studies,
    horus_qa_studies,
    isis_qa_studies,
    osiris_qa_studies,
    set_qa_studies,
    shu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17820


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_geb_qa_studies_family(seed: int = _SEED + 0):
    """geb_qa_studies: synthetic correctness bench."""
    return _finite_blob(geb_qa_studies.bench_geb_qa_studies(seed))


def bench_horus_qa_studies_family(seed: int = _SEED + 1):
    """horus_qa_studies: synthetic correctness bench."""
    return _finite_blob(horus_qa_studies.bench_horus_qa_studies(seed))


def bench_isis_qa_studies_family(seed: int = _SEED + 2):
    """isis_qa_studies: synthetic correctness bench."""
    return _finite_blob(isis_qa_studies.bench_isis_qa_studies(seed))


def bench_osiris_qa_studies_family(seed: int = _SEED + 3):
    """osiris_qa_studies: synthetic correctness bench."""
    return _finite_blob(osiris_qa_studies.bench_osiris_qa_studies(seed))


def bench_set_qa_studies_family(seed: int = _SEED + 4):
    """set_qa_studies: synthetic correctness bench."""
    return _finite_blob(set_qa_studies.bench_set_qa_studies(seed))


def bench_shu_qa_studies_family(seed: int = _SEED + 5):
    """shu_qa_studies: synthetic correctness bench."""
    return _finite_blob(shu_qa_studies.bench_shu_qa_studies(seed))
