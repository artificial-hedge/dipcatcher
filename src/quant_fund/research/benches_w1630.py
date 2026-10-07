"""Wave-1630 bench adapters: cryptid-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bigfoot_qa_studies,
    bunyip_qa_studies,
    loch_ness_qa_studies,
    rougarou_qa_studies,
    skinwalker_qa_studies,
    wendigo_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16300


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


def bench_bigfoot_qa_studies_family(seed: int = _SEED + 0):
    """bigfoot_qa_studies: synthetic correctness bench."""
    return _finite_blob(bigfoot_qa_studies.bench_bigfoot_qa_studies(seed))


def bench_bunyip_qa_studies_family(seed: int = _SEED + 1):
    """bunyip_qa_studies: synthetic correctness bench."""
    return _finite_blob(bunyip_qa_studies.bench_bunyip_qa_studies(seed))


def bench_loch_ness_qa_studies_family(seed: int = _SEED + 2):
    """loch_ness_qa_studies: synthetic correctness bench."""
    return _finite_blob(loch_ness_qa_studies.bench_loch_ness_qa_studies(seed))


def bench_rougarou_qa_studies_family(seed: int = _SEED + 3):
    """rougarou_qa_studies: synthetic correctness bench."""
    return _finite_blob(rougarou_qa_studies.bench_rougarou_qa_studies(seed))


def bench_skinwalker_qa_studies_family(seed: int = _SEED + 4):
    """skinwalker_qa_studies: synthetic correctness bench."""
    return _finite_blob(skinwalker_qa_studies.bench_skinwalker_qa_studies(seed))


def bench_wendigo_qa_studies_family(seed: int = _SEED + 5):
    """wendigo_qa_studies: synthetic correctness bench."""
    return _finite_blob(wendigo_qa_studies.bench_wendigo_qa_studies(seed))
