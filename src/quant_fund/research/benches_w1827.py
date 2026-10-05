"""Wave-1827 bench adapters: siberian-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    abaasy2_qa_studies,
    buga2_qa_studies,
    khosun2_qa_studies,
    kyys2_qa_studies,
    naa2_qa_studies,
    num2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18270


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_abaasy2_qa_studies_family(seed: int = _SEED + 0):
    """abaasy2_qa_studies: synthetic correctness bench."""
    return _finite_blob(abaasy2_qa_studies.bench_abaasy2_qa_studies(seed))


def bench_buga2_qa_studies_family(seed: int = _SEED + 1):
    """buga2_qa_studies: synthetic correctness bench."""
    return _finite_blob(buga2_qa_studies.bench_buga2_qa_studies(seed))


def bench_khosun2_qa_studies_family(seed: int = _SEED + 2):
    """khosun2_qa_studies: synthetic correctness bench."""
    return _finite_blob(khosun2_qa_studies.bench_khosun2_qa_studies(seed))


def bench_kyys2_qa_studies_family(seed: int = _SEED + 3):
    """kyys2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kyys2_qa_studies.bench_kyys2_qa_studies(seed))


def bench_naa2_qa_studies_family(seed: int = _SEED + 4):
    """naa2_qa_studies: synthetic correctness bench."""
    return _finite_blob(naa2_qa_studies.bench_naa2_qa_studies(seed))


def bench_num2_qa_studies_family(seed: int = _SEED + 5):
    """num2_qa_studies: synthetic correctness bench."""
    return _finite_blob(num2_qa_studies.bench_num2_qa_studies(seed))
