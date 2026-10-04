"""Wave-1499 bench adapters: spice-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    oregano_qa_studies,
    parsley_qa_studies,
    rosemary_qa_studies,
    saffron_qa_studies,
    tarragon_qa_studies,
    turmeric_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14990


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_oregano_qa_studies_family(seed: int = _SEED + 0):
    """oregano_qa_studies: synthetic correctness bench."""
    return _finite_blob(oregano_qa_studies.bench_oregano_qa_studies(seed))


def bench_parsley_qa_studies_family(seed: int = _SEED + 1):
    """parsley_qa_studies: synthetic correctness bench."""
    return _finite_blob(parsley_qa_studies.bench_parsley_qa_studies(seed))


def bench_rosemary_qa_studies_family(seed: int = _SEED + 2):
    """rosemary_qa_studies: synthetic correctness bench."""
    return _finite_blob(rosemary_qa_studies.bench_rosemary_qa_studies(seed))


def bench_saffron_qa_studies_family(seed: int = _SEED + 3):
    """saffron_qa_studies: synthetic correctness bench."""
    return _finite_blob(saffron_qa_studies.bench_saffron_qa_studies(seed))


def bench_tarragon_qa_studies_family(seed: int = _SEED + 4):
    """tarragon_qa_studies: synthetic correctness bench."""
    return _finite_blob(tarragon_qa_studies.bench_tarragon_qa_studies(seed))


def bench_turmeric_qa_studies_family(seed: int = _SEED + 5):
    """turmeric_qa_studies: synthetic correctness bench."""
    return _finite_blob(turmeric_qa_studies.bench_turmeric_qa_studies(seed))
