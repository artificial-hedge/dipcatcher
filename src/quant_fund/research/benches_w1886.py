"""Wave-1886 bench adapters: arabian-bestiary canon (SYNTHETIC only)."""

from quant_fund.models import (
    bahamut_qa_studies,
    falak_qa_studies,
    ghoula_qa_studies,
    karkadann_qa_studies,
    nasnas_qa_studies,
    shahmaran_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18860


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bahamut_qa_studies_family(seed: int = _SEED + 0):
    """bahamut_qa_studies: synthetic correctness bench."""
    return _finite_blob(bahamut_qa_studies.bench_bahamut_qa_studies(seed))


def bench_falak_qa_studies_family(seed: int = _SEED + 1):
    """falak_qa_studies: synthetic correctness bench."""
    return _finite_blob(falak_qa_studies.bench_falak_qa_studies(seed))


def bench_ghoula_qa_studies_family(seed: int = _SEED + 2):
    """ghoula_qa_studies: synthetic correctness bench."""
    return _finite_blob(ghoula_qa_studies.bench_ghoula_qa_studies(seed))


def bench_karkadann_qa_studies_family(seed: int = _SEED + 3):
    """karkadann_qa_studies: synthetic correctness bench."""
    return _finite_blob(karkadann_qa_studies.bench_karkadann_qa_studies(seed))


def bench_nasnas_qa_studies_family(seed: int = _SEED + 4):
    """nasnas_qa_studies: synthetic correctness bench."""
    return _finite_blob(nasnas_qa_studies.bench_nasnas_qa_studies(seed))


def bench_shahmaran_qa_studies_family(seed: int = _SEED + 5):
    """shahmaran_qa_studies: synthetic correctness bench."""
    return _finite_blob(shahmaran_qa_studies.bench_shahmaran_qa_studies(seed))
