"""Wave-1395 bench adapters: QA-exotics-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bamboogle_studies,
    fine_qa_studies,
    hotpot2_studies,
    kwik_qa_studies,
    quest_qa_studies,
    tatqa2_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13950


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bamboogle_studies_family(seed: int = _SEED + 0):
    """bamboogle_studies: synthetic correctness bench."""
    return _finite_blob(bamboogle_studies.bench_bamboogle_studies(seed))


def bench_fine_qa_studies_family(seed: int = _SEED + 1):
    """fine_qa_studies: synthetic correctness bench."""
    return _finite_blob(fine_qa_studies.bench_fine_qa_studies(seed))


def bench_hotpot2_studies_family(seed: int = _SEED + 2):
    """hotpot2_studies: synthetic correctness bench."""
    return _finite_blob(hotpot2_studies.bench_hotpot2_studies(seed))


def bench_kwik_qa_studies_family(seed: int = _SEED + 3):
    """kwik_qa_studies: synthetic correctness bench."""
    return _finite_blob(kwik_qa_studies.bench_kwik_qa_studies(seed))


def bench_quest_qa_studies_family(seed: int = _SEED + 4):
    """quest_qa_studies: synthetic correctness bench."""
    return _finite_blob(quest_qa_studies.bench_quest_qa_studies(seed))


def bench_tatqa2_studies_family(seed: int = _SEED + 5):
    """tatqa2_studies: synthetic correctness bench."""
    return _finite_blob(tatqa2_studies.bench_tatqa2_studies(seed))
