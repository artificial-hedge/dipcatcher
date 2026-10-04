"""Wave-1924 bench adapters: turkic-demon-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    achiyay_qa_studies,
    cayt_qa_studies,
    emegen_qa_studies,
    maymene_qa_studies,
    ubir_qa_studies,
    uor_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19240


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_achiyay_qa_studies_family(seed: int = _SEED + 0):
    """achiyay_qa_studies: synthetic correctness bench."""
    return _finite_blob(achiyay_qa_studies.bench_achiyay_qa_studies(seed))


def bench_cayt_qa_studies_family(seed: int = _SEED + 1):
    """cayt_qa_studies: synthetic correctness bench."""
    return _finite_blob(cayt_qa_studies.bench_cayt_qa_studies(seed))


def bench_emegen_qa_studies_family(seed: int = _SEED + 2):
    """emegen_qa_studies: synthetic correctness bench."""
    return _finite_blob(emegen_qa_studies.bench_emegen_qa_studies(seed))


def bench_maymene_qa_studies_family(seed: int = _SEED + 3):
    """maymene_qa_studies: synthetic correctness bench."""
    return _finite_blob(maymene_qa_studies.bench_maymene_qa_studies(seed))


def bench_ubir_qa_studies_family(seed: int = _SEED + 4):
    """ubir_qa_studies: synthetic correctness bench."""
    return _finite_blob(ubir_qa_studies.bench_ubir_qa_studies(seed))


def bench_uor_qa_studies_family(seed: int = _SEED + 5):
    """uor_qa_studies: synthetic correctness bench."""
    return _finite_blob(uor_qa_studies.bench_uor_qa_studies(seed))
