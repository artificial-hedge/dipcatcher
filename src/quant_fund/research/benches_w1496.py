"""Wave-1496 bench adapters: wader-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    jacana_qa_studies,
    lapwing_qa_studies,
    moorhen_qa_studies,
    railbird_qa_studies,
    snipe_qa_studies,
    turnstone_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14960


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_jacana_qa_studies_family(seed: int = _SEED + 0):
    """jacana_qa_studies: synthetic correctness bench."""
    return _finite_blob(jacana_qa_studies.bench_jacana_qa_studies(seed))


def bench_lapwing_qa_studies_family(seed: int = _SEED + 1):
    """lapwing_qa_studies: synthetic correctness bench."""
    return _finite_blob(lapwing_qa_studies.bench_lapwing_qa_studies(seed))


def bench_moorhen_qa_studies_family(seed: int = _SEED + 2):
    """moorhen_qa_studies: synthetic correctness bench."""
    return _finite_blob(moorhen_qa_studies.bench_moorhen_qa_studies(seed))


def bench_railbird_qa_studies_family(seed: int = _SEED + 3):
    """railbird_qa_studies: synthetic correctness bench."""
    return _finite_blob(railbird_qa_studies.bench_railbird_qa_studies(seed))


def bench_snipe_qa_studies_family(seed: int = _SEED + 4):
    """snipe_qa_studies: synthetic correctness bench."""
    return _finite_blob(snipe_qa_studies.bench_snipe_qa_studies(seed))


def bench_turnstone_qa_studies_family(seed: int = _SEED + 5):
    """turnstone_qa_studies: synthetic correctness bench."""
    return _finite_blob(turnstone_qa_studies.bench_turnstone_qa_studies(seed))
