"""Wave-1936 bench adapters: guarani-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    aoao_qa_studies,
    jasy_jatere_qa_studies,
    kurupi_qa_studies,
    luison_qa_studies,
    mboitui_qa_studies,
    pombero_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19360


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aoao_qa_studies_family(seed: int = _SEED + 0):
    """aoao_qa_studies: synthetic correctness bench."""
    return _finite_blob(aoao_qa_studies.bench_aoao_qa_studies(seed))


def bench_jasy_jatere_qa_studies_family(seed: int = _SEED + 1):
    """jasy_jatere_qa_studies: synthetic correctness bench."""
    return _finite_blob(jasy_jatere_qa_studies.bench_jasy_jatere_qa_studies(seed))


def bench_kurupi_qa_studies_family(seed: int = _SEED + 2):
    """kurupi_qa_studies: synthetic correctness bench."""
    return _finite_blob(kurupi_qa_studies.bench_kurupi_qa_studies(seed))


def bench_luison_qa_studies_family(seed: int = _SEED + 3):
    """luison_qa_studies: synthetic correctness bench."""
    return _finite_blob(luison_qa_studies.bench_luison_qa_studies(seed))


def bench_mboitui_qa_studies_family(seed: int = _SEED + 4):
    """mboitui_qa_studies: synthetic correctness bench."""
    return _finite_blob(mboitui_qa_studies.bench_mboitui_qa_studies(seed))


def bench_pombero_qa_studies_family(seed: int = _SEED + 5):
    """pombero_qa_studies: synthetic correctness bench."""
    return _finite_blob(pombero_qa_studies.bench_pombero_qa_studies(seed))
