"""Wave-1692 bench adapters: norse-myth-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    alfar_qa_studies,
    draugar_qa_studies,
    hulder_qa_studies,
    muspell_qa_studies,
    svartalf_qa_studies,
    ymir_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16920


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alfar_qa_studies_family(seed: int = _SEED + 0):
    """alfar_qa_studies: synthetic correctness bench."""
    return _finite_blob(alfar_qa_studies.bench_alfar_qa_studies(seed))


def bench_draugar_qa_studies_family(seed: int = _SEED + 1):
    """draugar_qa_studies: synthetic correctness bench."""
    return _finite_blob(draugar_qa_studies.bench_draugar_qa_studies(seed))


def bench_hulder_qa_studies_family(seed: int = _SEED + 2):
    """hulder_qa_studies: synthetic correctness bench."""
    return _finite_blob(hulder_qa_studies.bench_hulder_qa_studies(seed))


def bench_muspell_qa_studies_family(seed: int = _SEED + 3):
    """muspell_qa_studies: synthetic correctness bench."""
    return _finite_blob(muspell_qa_studies.bench_muspell_qa_studies(seed))


def bench_svartalf_qa_studies_family(seed: int = _SEED + 4):
    """svartalf_qa_studies: synthetic correctness bench."""
    return _finite_blob(svartalf_qa_studies.bench_svartalf_qa_studies(seed))


def bench_ymir_qa_studies_family(seed: int = _SEED + 5):
    """ymir_qa_studies: synthetic correctness bench."""
    return _finite_blob(ymir_qa_studies.bench_ymir_qa_studies(seed))
