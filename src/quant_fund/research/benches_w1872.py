"""Wave-1872 bench adapters: celtic-myth-6 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bugul_noz_qa_studies,
    cabyll_qa_studies,
    each_uisge_qa_studies,
    mooinjer_qa_studies,
    morveren_qa_studies,
    nuckelavee_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18720


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bugul_noz_qa_studies_family(seed: int = _SEED + 0):
    """bugul_noz_qa_studies: synthetic correctness bench."""
    return _finite_blob(bugul_noz_qa_studies.bench_bugul_noz_qa_studies(seed))


def bench_cabyll_qa_studies_family(seed: int = _SEED + 1):
    """cabyll_qa_studies: synthetic correctness bench."""
    return _finite_blob(cabyll_qa_studies.bench_cabyll_qa_studies(seed))


def bench_each_uisge_qa_studies_family(seed: int = _SEED + 2):
    """each_uisge_qa_studies: synthetic correctness bench."""
    return _finite_blob(each_uisge_qa_studies.bench_each_uisge_qa_studies(seed))


def bench_mooinjer_qa_studies_family(seed: int = _SEED + 3):
    """mooinjer_qa_studies: synthetic correctness bench."""
    return _finite_blob(mooinjer_qa_studies.bench_mooinjer_qa_studies(seed))


def bench_morveren_qa_studies_family(seed: int = _SEED + 4):
    """morveren_qa_studies: synthetic correctness bench."""
    return _finite_blob(morveren_qa_studies.bench_morveren_qa_studies(seed))


def bench_nuckelavee_qa_studies_family(seed: int = _SEED + 5):
    """nuckelavee_qa_studies: synthetic correctness bench."""
    return _finite_blob(nuckelavee_qa_studies.bench_nuckelavee_qa_studies(seed))
