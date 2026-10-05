"""Wave-1908 bench adapters: brazilian-slavic remnant canon (SYNTHETIC only)."""

from quant_fund.models import (
    anhanga_qa_studies,
    bolotnik_qa_studies,
    dvorovoy_qa_studies,
    jurupari_qa_studies,
    lobisomem_qa_studies,
    mula_sem_cabeca_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19080


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_anhanga_qa_studies_family(seed: int = _SEED + 0):
    """anhanga_qa_studies: synthetic correctness bench."""
    return _finite_blob(anhanga_qa_studies.bench_anhanga_qa_studies(seed))


def bench_bolotnik_qa_studies_family(seed: int = _SEED + 1):
    """bolotnik_qa_studies: synthetic correctness bench."""
    return _finite_blob(bolotnik_qa_studies.bench_bolotnik_qa_studies(seed))


def bench_dvorovoy_qa_studies_family(seed: int = _SEED + 2):
    """dvorovoy_qa_studies: synthetic correctness bench."""
    return _finite_blob(dvorovoy_qa_studies.bench_dvorovoy_qa_studies(seed))


def bench_jurupari_qa_studies_family(seed: int = _SEED + 3):
    """jurupari_qa_studies: synthetic correctness bench."""
    return _finite_blob(jurupari_qa_studies.bench_jurupari_qa_studies(seed))


def bench_lobisomem_qa_studies_family(seed: int = _SEED + 4):
    """lobisomem_qa_studies: synthetic correctness bench."""
    return _finite_blob(lobisomem_qa_studies.bench_lobisomem_qa_studies(seed))


def bench_mula_sem_cabeca_qa_studies_family(seed: int = _SEED + 5):
    """mula_sem_cabeca_qa_studies: synthetic correctness bench."""
    return _finite_blob(mula_sem_cabeca_qa_studies.bench_mula_sem_cabeca_qa_studies(seed))
