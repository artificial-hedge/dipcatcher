"""Wave-1514 bench adapters: butterfly canon (SYNTHETIC only)."""

from quant_fund.models import (
    blue_morpho_qa_studies,
    cabbage_white_qa_studies,
    fritillary_qa_studies,
    monarch_qa_studies,
    painted_lady_qa_studies,
    swallowtail_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15140


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_blue_morpho_qa_studies_family(seed: int = _SEED + 0):
    """blue_morpho_qa_studies: synthetic correctness bench."""
    return _finite_blob(blue_morpho_qa_studies.bench_blue_morpho_qa_studies(seed))


def bench_cabbage_white_qa_studies_family(seed: int = _SEED + 1):
    """cabbage_white_qa_studies: synthetic correctness bench."""
    return _finite_blob(cabbage_white_qa_studies.bench_cabbage_white_qa_studies(seed))


def bench_fritillary_qa_studies_family(seed: int = _SEED + 2):
    """fritillary_qa_studies: synthetic correctness bench."""
    return _finite_blob(fritillary_qa_studies.bench_fritillary_qa_studies(seed))


def bench_monarch_qa_studies_family(seed: int = _SEED + 3):
    """monarch_qa_studies: synthetic correctness bench."""
    return _finite_blob(monarch_qa_studies.bench_monarch_qa_studies(seed))


def bench_painted_lady_qa_studies_family(seed: int = _SEED + 4):
    """painted_lady_qa_studies: synthetic correctness bench."""
    return _finite_blob(painted_lady_qa_studies.bench_painted_lady_qa_studies(seed))


def bench_swallowtail_qa_studies_family(seed: int = _SEED + 5):
    """swallowtail_qa_studies: synthetic correctness bench."""
    return _finite_blob(swallowtail_qa_studies.bench_swallowtail_qa_studies(seed))
