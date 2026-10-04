"""Wave-1344 bench adapters: reading-comprehension-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    boolq_qa_studies,
    dream_qa_studies,
    duorc_qa_studies,
    mctest_qa_studies,
    qasper_qa_studies,
    race_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13440


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_boolq_qa_studies_family(seed: int = _SEED + 0):
    """boolq_qa_studies: synthetic correctness bench."""
    return _finite_blob(boolq_qa_studies.bench_boolq_qa_studies(seed))


def bench_dream_qa_studies_family(seed: int = _SEED + 1):
    """dream_qa_studies: synthetic correctness bench."""
    return _finite_blob(dream_qa_studies.bench_dream_qa_studies(seed))


def bench_duorc_qa_studies_family(seed: int = _SEED + 2):
    """duorc_qa_studies: synthetic correctness bench."""
    return _finite_blob(duorc_qa_studies.bench_duorc_qa_studies(seed))


def bench_mctest_qa_studies_family(seed: int = _SEED + 3):
    """mctest_qa_studies: synthetic correctness bench."""
    return _finite_blob(mctest_qa_studies.bench_mctest_qa_studies(seed))


def bench_qasper_qa_studies_family(seed: int = _SEED + 4):
    """qasper_qa_studies: synthetic correctness bench."""
    return _finite_blob(qasper_qa_studies.bench_qasper_qa_studies(seed))


def bench_race_qa_studies_family(seed: int = _SEED + 5):
    """race_qa_studies: synthetic correctness bench."""
    return _finite_blob(race_qa_studies.bench_race_qa_studies(seed))
