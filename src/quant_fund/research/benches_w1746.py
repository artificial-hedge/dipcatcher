"""Wave-1746 bench adapters: sumerian-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    ishtar_qa_studies,
    nanna_qa_studies,
    nergal_qa_studies,
    ninurta_qa_studies,
    nisaba_qa_studies,
    utu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17460


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ishtar_qa_studies_family(seed: int = _SEED + 0):
    """ishtar_qa_studies: synthetic correctness bench."""
    return _finite_blob(ishtar_qa_studies.bench_ishtar_qa_studies(seed))


def bench_nanna_qa_studies_family(seed: int = _SEED + 1):
    """nanna_qa_studies: synthetic correctness bench."""
    return _finite_blob(nanna_qa_studies.bench_nanna_qa_studies(seed))


def bench_nergal_qa_studies_family(seed: int = _SEED + 2):
    """nergal_qa_studies: synthetic correctness bench."""
    return _finite_blob(nergal_qa_studies.bench_nergal_qa_studies(seed))


def bench_ninurta_qa_studies_family(seed: int = _SEED + 3):
    """ninurta_qa_studies: synthetic correctness bench."""
    return _finite_blob(ninurta_qa_studies.bench_ninurta_qa_studies(seed))


def bench_nisaba_qa_studies_family(seed: int = _SEED + 4):
    """nisaba_qa_studies: synthetic correctness bench."""
    return _finite_blob(nisaba_qa_studies.bench_nisaba_qa_studies(seed))


def bench_utu_qa_studies_family(seed: int = _SEED + 5):
    """utu_qa_studies: synthetic correctness bench."""
    return _finite_blob(utu_qa_studies.bench_utu_qa_studies(seed))
