"""Wave-1653 bench adapters: global-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    alion_qa_studies,
    catoblepas_qa_studies,
    jasconius_qa_studies,
    pard_qa_studies,
    peluda_qa_studies,
    zaratan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16530


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alion_qa_studies_family(seed: int = _SEED + 0):
    """alion_qa_studies: synthetic correctness bench."""
    return _finite_blob(alion_qa_studies.bench_alion_qa_studies(seed))


def bench_catoblepas_qa_studies_family(seed: int = _SEED + 1):
    """catoblepas_qa_studies: synthetic correctness bench."""
    return _finite_blob(catoblepas_qa_studies.bench_catoblepas_qa_studies(seed))


def bench_jasconius_qa_studies_family(seed: int = _SEED + 2):
    """jasconius_qa_studies: synthetic correctness bench."""
    return _finite_blob(jasconius_qa_studies.bench_jasconius_qa_studies(seed))


def bench_pard_qa_studies_family(seed: int = _SEED + 3):
    """pard_qa_studies: synthetic correctness bench."""
    return _finite_blob(pard_qa_studies.bench_pard_qa_studies(seed))


def bench_peluda_qa_studies_family(seed: int = _SEED + 4):
    """peluda_qa_studies: synthetic correctness bench."""
    return _finite_blob(peluda_qa_studies.bench_peluda_qa_studies(seed))


def bench_zaratan_qa_studies_family(seed: int = _SEED + 5):
    """zaratan_qa_studies: synthetic correctness bench."""
    return _finite_blob(zaratan_qa_studies.bench_zaratan_qa_studies(seed))
