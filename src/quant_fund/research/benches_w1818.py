"""Wave-1818 bench adapters: egyptian-9 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bastet2_qa_studies,
    geb2_qa_studies,
    hathor2_qa_studies,
    nut2_qa_studies,
    sekhmet2_qa_studies,
    tefnut2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18180


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bastet2_qa_studies_family(seed: int = _SEED + 0):
    """bastet2_qa_studies: synthetic correctness bench."""
    return _finite_blob(bastet2_qa_studies.bench_bastet2_qa_studies(seed))


def bench_geb2_qa_studies_family(seed: int = _SEED + 1):
    """geb2_qa_studies: synthetic correctness bench."""
    return _finite_blob(geb2_qa_studies.bench_geb2_qa_studies(seed))


def bench_hathor2_qa_studies_family(seed: int = _SEED + 2):
    """hathor2_qa_studies: synthetic correctness bench."""
    return _finite_blob(hathor2_qa_studies.bench_hathor2_qa_studies(seed))


def bench_nut2_qa_studies_family(seed: int = _SEED + 3):
    """nut2_qa_studies: synthetic correctness bench."""
    return _finite_blob(nut2_qa_studies.bench_nut2_qa_studies(seed))


def bench_sekhmet2_qa_studies_family(seed: int = _SEED + 4):
    """sekhmet2_qa_studies: synthetic correctness bench."""
    return _finite_blob(sekhmet2_qa_studies.bench_sekhmet2_qa_studies(seed))


def bench_tefnut2_qa_studies_family(seed: int = _SEED + 5):
    """tefnut2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tefnut2_qa_studies.bench_tefnut2_qa_studies(seed))
