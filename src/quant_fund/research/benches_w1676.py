"""Wave-1676 bench adapters: hindu-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    kinnara_qa_studies,
    pisacha_qa_studies,
    uraga_qa_studies,
    vetala_qa_studies,
    vidyadhara_qa_studies,
    yakshini_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16760


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_kinnara_qa_studies_family(seed: int = _SEED + 0):
    """kinnara_qa_studies: synthetic correctness bench."""
    return _finite_blob(kinnara_qa_studies.bench_kinnara_qa_studies(seed))


def bench_pisacha_qa_studies_family(seed: int = _SEED + 1):
    """pisacha_qa_studies: synthetic correctness bench."""
    return _finite_blob(pisacha_qa_studies.bench_pisacha_qa_studies(seed))


def bench_uraga_qa_studies_family(seed: int = _SEED + 2):
    """uraga_qa_studies: synthetic correctness bench."""
    return _finite_blob(uraga_qa_studies.bench_uraga_qa_studies(seed))


def bench_vetala_qa_studies_family(seed: int = _SEED + 3):
    """vetala_qa_studies: synthetic correctness bench."""
    return _finite_blob(vetala_qa_studies.bench_vetala_qa_studies(seed))


def bench_vidyadhara_qa_studies_family(seed: int = _SEED + 4):
    """vidyadhara_qa_studies: synthetic correctness bench."""
    return _finite_blob(vidyadhara_qa_studies.bench_vidyadhara_qa_studies(seed))


def bench_yakshini_qa_studies_family(seed: int = _SEED + 5):
    """yakshini_qa_studies: synthetic correctness bench."""
    return _finite_blob(yakshini_qa_studies.bench_yakshini_qa_studies(seed))
