"""Wave-1919 bench adapters: celtic-demon-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bodach_qa_studies,
    caointeach_qa_studies,
    fachan_qa_studies,
    geancanach_qa_studies,
    kilmoulis_qa_studies,
    shellycoat_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19190


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bodach_qa_studies_family(seed: int = _SEED + 0):
    """bodach_qa_studies: synthetic correctness bench."""
    return _finite_blob(bodach_qa_studies.bench_bodach_qa_studies(seed))


def bench_caointeach_qa_studies_family(seed: int = _SEED + 1):
    """caointeach_qa_studies: synthetic correctness bench."""
    return _finite_blob(caointeach_qa_studies.bench_caointeach_qa_studies(seed))


def bench_fachan_qa_studies_family(seed: int = _SEED + 2):
    """fachan_qa_studies: synthetic correctness bench."""
    return _finite_blob(fachan_qa_studies.bench_fachan_qa_studies(seed))


def bench_geancanach_qa_studies_family(seed: int = _SEED + 3):
    """geancanach_qa_studies: synthetic correctness bench."""
    return _finite_blob(geancanach_qa_studies.bench_geancanach_qa_studies(seed))


def bench_kilmoulis_qa_studies_family(seed: int = _SEED + 4):
    """kilmoulis_qa_studies: synthetic correctness bench."""
    return _finite_blob(kilmoulis_qa_studies.bench_kilmoulis_qa_studies(seed))


def bench_shellycoat_qa_studies_family(seed: int = _SEED + 5):
    """shellycoat_qa_studies: synthetic correctness bench."""
    return _finite_blob(shellycoat_qa_studies.bench_shellycoat_qa_studies(seed))
