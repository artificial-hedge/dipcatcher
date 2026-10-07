"""Wave-1589 bench adapters: cetacean-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    porpoise_qa_studies,
    right_whale_qa_studies,
    rissos_qa_studies,
    river_dolphin_qa_studies,
    spinner_qa_studies,
    vaquita_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15890


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


def bench_porpoise_qa_studies_family(seed: int = _SEED + 0):
    """porpoise_qa_studies: synthetic correctness bench."""
    return _finite_blob(porpoise_qa_studies.bench_porpoise_qa_studies(seed))


def bench_right_whale_qa_studies_family(seed: int = _SEED + 1):
    """right_whale_qa_studies: synthetic correctness bench."""
    return _finite_blob(right_whale_qa_studies.bench_right_whale_qa_studies(seed))


def bench_rissos_qa_studies_family(seed: int = _SEED + 2):
    """rissos_qa_studies: synthetic correctness bench."""
    return _finite_blob(rissos_qa_studies.bench_rissos_qa_studies(seed))


def bench_river_dolphin_qa_studies_family(seed: int = _SEED + 3):
    """river_dolphin_qa_studies: synthetic correctness bench."""
    return _finite_blob(river_dolphin_qa_studies.bench_river_dolphin_qa_studies(seed))


def bench_spinner_qa_studies_family(seed: int = _SEED + 4):
    """spinner_qa_studies: synthetic correctness bench."""
    return _finite_blob(spinner_qa_studies.bench_spinner_qa_studies(seed))


def bench_vaquita_qa_studies_family(seed: int = _SEED + 5):
    """vaquita_qa_studies: synthetic correctness bench."""
    return _finite_blob(vaquita_qa_studies.bench_vaquita_qa_studies(seed))
