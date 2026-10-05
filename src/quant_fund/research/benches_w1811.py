"""Wave-1811 bench adapters: egyptian-8 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bes2_qa_studies,
    geb2_qa_studies,
    nephthys2_qa_studies,
    ptah2_qa_studies,
    sekhmet2_qa_studies,
    thoth2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18110


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bes2_qa_studies_family(seed: int = _SEED + 0):
    """bes2_qa_studies: synthetic correctness bench."""
    return _finite_blob(bes2_qa_studies.bench_bes2_qa_studies(seed))


def bench_geb2_qa_studies_family(seed: int = _SEED + 1):
    """geb2_qa_studies: synthetic correctness bench."""
    return _finite_blob(geb2_qa_studies.bench_geb2_qa_studies(seed))


def bench_nephthys2_qa_studies_family(seed: int = _SEED + 2):
    """nephthys2_qa_studies: synthetic correctness bench."""
    return _finite_blob(nephthys2_qa_studies.bench_nephthys2_qa_studies(seed))


def bench_ptah2_qa_studies_family(seed: int = _SEED + 3):
    """ptah2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ptah2_qa_studies.bench_ptah2_qa_studies(seed))


def bench_sekhmet2_qa_studies_family(seed: int = _SEED + 4):
    """sekhmet2_qa_studies: synthetic correctness bench."""
    return _finite_blob(sekhmet2_qa_studies.bench_sekhmet2_qa_studies(seed))


def bench_thoth2_qa_studies_family(seed: int = _SEED + 5):
    """thoth2_qa_studies: synthetic correctness bench."""
    return _finite_blob(thoth2_qa_studies.bench_thoth2_qa_studies(seed))
