"""Wave-1814 bench adapters: mesopotamian-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    enki2_qa_studies,
    marduk2_qa_studies,
    nanna2_qa_studies,
    ninhursag2_qa_studies,
    tiamat2_qa_studies,
    utu2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18140


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_enki2_qa_studies_family(seed: int = _SEED + 0):
    """enki2_qa_studies: synthetic correctness bench."""
    return _finite_blob(enki2_qa_studies.bench_enki2_qa_studies(seed))


def bench_marduk2_qa_studies_family(seed: int = _SEED + 1):
    """marduk2_qa_studies: synthetic correctness bench."""
    return _finite_blob(marduk2_qa_studies.bench_marduk2_qa_studies(seed))


def bench_nanna2_qa_studies_family(seed: int = _SEED + 2):
    """nanna2_qa_studies: synthetic correctness bench."""
    return _finite_blob(nanna2_qa_studies.bench_nanna2_qa_studies(seed))


def bench_ninhursag2_qa_studies_family(seed: int = _SEED + 3):
    """ninhursag2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ninhursag2_qa_studies.bench_ninhursag2_qa_studies(seed))


def bench_tiamat2_qa_studies_family(seed: int = _SEED + 4):
    """tiamat2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tiamat2_qa_studies.bench_tiamat2_qa_studies(seed))


def bench_utu2_qa_studies_family(seed: int = _SEED + 5):
    """utu2_qa_studies: synthetic correctness bench."""
    return _finite_blob(utu2_qa_studies.bench_utu2_qa_studies(seed))
