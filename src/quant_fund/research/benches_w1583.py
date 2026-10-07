"""Wave-1583 bench adapters: cetacean canon (SYNTHETIC only)."""

from quant_fund.models import (
    bowhead_qa_studies,
    fin_whale_qa_studies,
    humpback_qa_studies,
    minke_qa_studies,
    pilot_whale_qa_studies,
    sperm_whale_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15830


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


def bench_bowhead_qa_studies_family(seed: int = _SEED + 0):
    """bowhead_qa_studies: synthetic correctness bench."""
    return _finite_blob(bowhead_qa_studies.bench_bowhead_qa_studies(seed))


def bench_fin_whale_qa_studies_family(seed: int = _SEED + 1):
    """fin_whale_qa_studies: synthetic correctness bench."""
    return _finite_blob(fin_whale_qa_studies.bench_fin_whale_qa_studies(seed))


def bench_humpback_qa_studies_family(seed: int = _SEED + 2):
    """humpback_qa_studies: synthetic correctness bench."""
    return _finite_blob(humpback_qa_studies.bench_humpback_qa_studies(seed))


def bench_minke_qa_studies_family(seed: int = _SEED + 3):
    """minke_qa_studies: synthetic correctness bench."""
    return _finite_blob(minke_qa_studies.bench_minke_qa_studies(seed))


def bench_pilot_whale_qa_studies_family(seed: int = _SEED + 4):
    """pilot_whale_qa_studies: synthetic correctness bench."""
    return _finite_blob(pilot_whale_qa_studies.bench_pilot_whale_qa_studies(seed))


def bench_sperm_whale_qa_studies_family(seed: int = _SEED + 5):
    """sperm_whale_qa_studies: synthetic correctness bench."""
    return _finite_blob(sperm_whale_qa_studies.bench_sperm_whale_qa_studies(seed))
