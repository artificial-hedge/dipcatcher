"""Wave-1636 bench adapters: yokai-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    gashadokuro_qa_studies,
    jorogumo_qa_studies,
    kodama_qa_studies,
    namahage_qa_studies,
    nue_2_qa_studies,
    tsuchinoko_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16360


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_gashadokuro_qa_studies_family(seed: int = _SEED + 0):
    """gashadokuro_qa_studies: synthetic correctness bench."""
    return _finite_blob(gashadokuro_qa_studies.bench_gashadokuro_qa_studies(seed))


def bench_jorogumo_qa_studies_family(seed: int = _SEED + 1):
    """jorogumo_qa_studies: synthetic correctness bench."""
    return _finite_blob(jorogumo_qa_studies.bench_jorogumo_qa_studies(seed))


def bench_kodama_qa_studies_family(seed: int = _SEED + 2):
    """kodama_qa_studies: synthetic correctness bench."""
    return _finite_blob(kodama_qa_studies.bench_kodama_qa_studies(seed))


def bench_namahage_qa_studies_family(seed: int = _SEED + 3):
    """namahage_qa_studies: synthetic correctness bench."""
    return _finite_blob(namahage_qa_studies.bench_namahage_qa_studies(seed))


def bench_nue_2_qa_studies_family(seed: int = _SEED + 4):
    """nue_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(nue_2_qa_studies.bench_nue_2_qa_studies(seed))


def bench_tsuchinoko_qa_studies_family(seed: int = _SEED + 5):
    """tsuchinoko_qa_studies: synthetic correctness bench."""
    return _finite_blob(tsuchinoko_qa_studies.bench_tsuchinoko_qa_studies(seed))
