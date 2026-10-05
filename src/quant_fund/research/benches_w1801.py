"""Wave-1801 bench adapters: incan-myth-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    apocatequil2_qa_studies,
    inti2_qa_studies,
    kon2_qa_studies,
    pacamama2_qa_studies,
    supay2_qa_studies,
    viracocha2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18010


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_apocatequil2_qa_studies_family(seed: int = _SEED + 0):
    """apocatequil2_qa_studies: synthetic correctness bench."""
    return _finite_blob(apocatequil2_qa_studies.bench_apocatequil2_qa_studies(seed))


def bench_inti2_qa_studies_family(seed: int = _SEED + 1):
    """inti2_qa_studies: synthetic correctness bench."""
    return _finite_blob(inti2_qa_studies.bench_inti2_qa_studies(seed))


def bench_kon2_qa_studies_family(seed: int = _SEED + 2):
    """kon2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kon2_qa_studies.bench_kon2_qa_studies(seed))


def bench_pacamama2_qa_studies_family(seed: int = _SEED + 3):
    """pacamama2_qa_studies: synthetic correctness bench."""
    return _finite_blob(pacamama2_qa_studies.bench_pacamama2_qa_studies(seed))


def bench_supay2_qa_studies_family(seed: int = _SEED + 4):
    """supay2_qa_studies: synthetic correctness bench."""
    return _finite_blob(supay2_qa_studies.bench_supay2_qa_studies(seed))


def bench_viracocha2_qa_studies_family(seed: int = _SEED + 5):
    """viracocha2_qa_studies: synthetic correctness bench."""
    return _finite_blob(viracocha2_qa_studies.bench_viracocha2_qa_studies(seed))
