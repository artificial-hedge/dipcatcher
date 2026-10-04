"""Wave-1703 bench adapters: sumerian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    dumuzi_qa_studies,
    inanna_qa_studies,
    marduk_qa_studies,
    namtar_qa_studies,
    nergal_qa_studies,
    ninhursag_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17030


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_dumuzi_qa_studies_family(seed: int = _SEED + 0):
    """dumuzi_qa_studies: synthetic correctness bench."""
    return _finite_blob(dumuzi_qa_studies.bench_dumuzi_qa_studies(seed))


def bench_inanna_qa_studies_family(seed: int = _SEED + 1):
    """inanna_qa_studies: synthetic correctness bench."""
    return _finite_blob(inanna_qa_studies.bench_inanna_qa_studies(seed))


def bench_marduk_qa_studies_family(seed: int = _SEED + 2):
    """marduk_qa_studies: synthetic correctness bench."""
    return _finite_blob(marduk_qa_studies.bench_marduk_qa_studies(seed))


def bench_namtar_qa_studies_family(seed: int = _SEED + 3):
    """namtar_qa_studies: synthetic correctness bench."""
    return _finite_blob(namtar_qa_studies.bench_namtar_qa_studies(seed))


def bench_nergal_qa_studies_family(seed: int = _SEED + 4):
    """nergal_qa_studies: synthetic correctness bench."""
    return _finite_blob(nergal_qa_studies.bench_nergal_qa_studies(seed))


def bench_ninhursag_qa_studies_family(seed: int = _SEED + 5):
    """ninhursag_qa_studies: synthetic correctness bench."""
    return _finite_blob(ninhursag_qa_studies.bench_ninhursag_qa_studies(seed))
