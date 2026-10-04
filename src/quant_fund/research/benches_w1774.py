"""Wave-1774 bench adapters: egyptian-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bastet_qa_studies,
    hathor_qa_studies,
    nut_qa_studies,
    sekhmet_qa_studies,
    sobek_qa_studies,
    thoth_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17740


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bastet_qa_studies_family(seed: int = _SEED + 0):
    """bastet_qa_studies: synthetic correctness bench."""
    return _finite_blob(bastet_qa_studies.bench_bastet_qa_studies(seed))


def bench_hathor_qa_studies_family(seed: int = _SEED + 1):
    """hathor_qa_studies: synthetic correctness bench."""
    return _finite_blob(hathor_qa_studies.bench_hathor_qa_studies(seed))


def bench_nut_qa_studies_family(seed: int = _SEED + 2):
    """nut_qa_studies: synthetic correctness bench."""
    return _finite_blob(nut_qa_studies.bench_nut_qa_studies(seed))


def bench_sekhmet_qa_studies_family(seed: int = _SEED + 3):
    """sekhmet_qa_studies: synthetic correctness bench."""
    return _finite_blob(sekhmet_qa_studies.bench_sekhmet_qa_studies(seed))


def bench_sobek_qa_studies_family(seed: int = _SEED + 4):
    """sobek_qa_studies: synthetic correctness bench."""
    return _finite_blob(sobek_qa_studies.bench_sobek_qa_studies(seed))


def bench_thoth_qa_studies_family(seed: int = _SEED + 5):
    """thoth_qa_studies: synthetic correctness bench."""
    return _finite_blob(thoth_qa_studies.bench_thoth_qa_studies(seed))
