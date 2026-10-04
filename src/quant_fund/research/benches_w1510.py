"""Wave-1510 bench adapters: corvid canon (SYNTHETIC only)."""

from quant_fund.models import (
    chough_qa_studies,
    crow_qa_studies,
    jackdaw_qa_studies,
    jay_qa_studies,
    magpie_qa_studies,
    rook_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15100


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_chough_qa_studies_family(seed: int = _SEED + 0):
    """chough_qa_studies: synthetic correctness bench."""
    return _finite_blob(chough_qa_studies.bench_chough_qa_studies(seed))


def bench_crow_qa_studies_family(seed: int = _SEED + 1):
    """crow_qa_studies: synthetic correctness bench."""
    return _finite_blob(crow_qa_studies.bench_crow_qa_studies(seed))


def bench_jackdaw_qa_studies_family(seed: int = _SEED + 2):
    """jackdaw_qa_studies: synthetic correctness bench."""
    return _finite_blob(jackdaw_qa_studies.bench_jackdaw_qa_studies(seed))


def bench_jay_qa_studies_family(seed: int = _SEED + 3):
    """jay_qa_studies: synthetic correctness bench."""
    return _finite_blob(jay_qa_studies.bench_jay_qa_studies(seed))


def bench_magpie_qa_studies_family(seed: int = _SEED + 4):
    """magpie_qa_studies: synthetic correctness bench."""
    return _finite_blob(magpie_qa_studies.bench_magpie_qa_studies(seed))


def bench_rook_qa_studies_family(seed: int = _SEED + 5):
    """rook_qa_studies: synthetic correctness bench."""
    return _finite_blob(rook_qa_studies.bench_rook_qa_studies(seed))
