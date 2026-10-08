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
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
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
