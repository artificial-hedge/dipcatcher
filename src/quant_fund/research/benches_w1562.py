"""Wave-1562 bench adapters: reef-fish-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    boxfish_qa_studies,
    clownfish_qa_studies,
    dragonet_qa_studies,
    mandarinfish_qa_studies,
    pipefish_qa_studies,
    pufferfish_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15620


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


def bench_boxfish_qa_studies_family(seed: int = _SEED + 0):
    """boxfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(boxfish_qa_studies.bench_boxfish_qa_studies(seed))


def bench_clownfish_qa_studies_family(seed: int = _SEED + 1):
    """clownfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(clownfish_qa_studies.bench_clownfish_qa_studies(seed))


def bench_dragonet_qa_studies_family(seed: int = _SEED + 2):
    """dragonet_qa_studies: synthetic correctness bench."""
    return _finite_blob(dragonet_qa_studies.bench_dragonet_qa_studies(seed))


def bench_mandarinfish_qa_studies_family(seed: int = _SEED + 3):
    """mandarinfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(mandarinfish_qa_studies.bench_mandarinfish_qa_studies(seed))


def bench_pipefish_qa_studies_family(seed: int = _SEED + 4):
    """pipefish_qa_studies: synthetic correctness bench."""
    return _finite_blob(pipefish_qa_studies.bench_pipefish_qa_studies(seed))


def bench_pufferfish_qa_studies_family(seed: int = _SEED + 5):
    """pufferfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(pufferfish_qa_studies.bench_pufferfish_qa_studies(seed))
