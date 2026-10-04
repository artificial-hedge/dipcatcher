"""Wave-1534 bench adapters: nightbird canon (SYNTHETIC only)."""

from quant_fund.models import (
    cuckoo_qa_studies,
    frogmouth_qa_studies,
    koel_qa_studies,
    nighthawk_qa_studies,
    nightjar_qa_studies,
    roadrunner_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15340


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cuckoo_qa_studies_family(seed: int = _SEED + 0):
    """cuckoo_qa_studies: synthetic correctness bench."""
    return _finite_blob(cuckoo_qa_studies.bench_cuckoo_qa_studies(seed))


def bench_frogmouth_qa_studies_family(seed: int = _SEED + 1):
    """frogmouth_qa_studies: synthetic correctness bench."""
    return _finite_blob(frogmouth_qa_studies.bench_frogmouth_qa_studies(seed))


def bench_koel_qa_studies_family(seed: int = _SEED + 2):
    """koel_qa_studies: synthetic correctness bench."""
    return _finite_blob(koel_qa_studies.bench_koel_qa_studies(seed))


def bench_nighthawk_qa_studies_family(seed: int = _SEED + 3):
    """nighthawk_qa_studies: synthetic correctness bench."""
    return _finite_blob(nighthawk_qa_studies.bench_nighthawk_qa_studies(seed))


def bench_nightjar_qa_studies_family(seed: int = _SEED + 4):
    """nightjar_qa_studies: synthetic correctness bench."""
    return _finite_blob(nightjar_qa_studies.bench_nightjar_qa_studies(seed))


def bench_roadrunner_qa_studies_family(seed: int = _SEED + 5):
    """roadrunner_qa_studies: synthetic correctness bench."""
    return _finite_blob(roadrunner_qa_studies.bench_roadrunner_qa_studies(seed))
