"""Wave-1763 bench adapters: egyptian-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    hapi_qa_studies,
    khnum_qa_studies,
    menhit_qa_studies,
    nephthys_qa_studies,
    serqet_qa_studies,
    tefnut_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17630


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_hapi_qa_studies_family(seed: int = _SEED + 0):
    """hapi_qa_studies: synthetic correctness bench."""
    return _finite_blob(hapi_qa_studies.bench_hapi_qa_studies(seed))


def bench_khnum_qa_studies_family(seed: int = _SEED + 1):
    """khnum_qa_studies: synthetic correctness bench."""
    return _finite_blob(khnum_qa_studies.bench_khnum_qa_studies(seed))


def bench_menhit_qa_studies_family(seed: int = _SEED + 2):
    """menhit_qa_studies: synthetic correctness bench."""
    return _finite_blob(menhit_qa_studies.bench_menhit_qa_studies(seed))


def bench_nephthys_qa_studies_family(seed: int = _SEED + 3):
    """nephthys_qa_studies: synthetic correctness bench."""
    return _finite_blob(nephthys_qa_studies.bench_nephthys_qa_studies(seed))


def bench_serqet_qa_studies_family(seed: int = _SEED + 4):
    """serqet_qa_studies: synthetic correctness bench."""
    return _finite_blob(serqet_qa_studies.bench_serqet_qa_studies(seed))


def bench_tefnut_qa_studies_family(seed: int = _SEED + 5):
    """tefnut_qa_studies: synthetic correctness bench."""
    return _finite_blob(tefnut_qa_studies.bench_tefnut_qa_studies(seed))
