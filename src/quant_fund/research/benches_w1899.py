"""Wave-1899 bench adapters: hindu-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    aghasura_qa_studies,
    bakasura_qa_studies,
    daitya_qa_studies,
    diti_qa_studies,
    pishacha_qa_studies,
    putana_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18990


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aghasura_qa_studies_family(seed: int = _SEED + 0):
    """aghasura_qa_studies: synthetic correctness bench."""
    return _finite_blob(aghasura_qa_studies.bench_aghasura_qa_studies(seed))


def bench_bakasura_qa_studies_family(seed: int = _SEED + 1):
    """bakasura_qa_studies: synthetic correctness bench."""
    return _finite_blob(bakasura_qa_studies.bench_bakasura_qa_studies(seed))


def bench_daitya_qa_studies_family(seed: int = _SEED + 2):
    """daitya_qa_studies: synthetic correctness bench."""
    return _finite_blob(daitya_qa_studies.bench_daitya_qa_studies(seed))


def bench_diti_qa_studies_family(seed: int = _SEED + 3):
    """diti_qa_studies: synthetic correctness bench."""
    return _finite_blob(diti_qa_studies.bench_diti_qa_studies(seed))


def bench_pishacha_qa_studies_family(seed: int = _SEED + 4):
    """pishacha_qa_studies: synthetic correctness bench."""
    return _finite_blob(pishacha_qa_studies.bench_pishacha_qa_studies(seed))


def bench_putana_qa_studies_family(seed: int = _SEED + 5):
    """putana_qa_studies: synthetic correctness bench."""
    return _finite_blob(putana_qa_studies.bench_putana_qa_studies(seed))
