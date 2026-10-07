"""Wave-1606 bench adapters: camelid-steppe canon (SYNTHETIC only)."""

from quant_fund.models import (
    alpaca_qa_studies,
    aoudad_qa_studies,
    dromedary_qa_studies,
    guanaco_qa_studies,
    salt_qa_studies,
    vicuna_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16060


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


def bench_alpaca_qa_studies_family(seed: int = _SEED + 0):
    """alpaca_qa_studies: synthetic correctness bench."""
    return _finite_blob(alpaca_qa_studies.bench_alpaca_qa_studies(seed))


def bench_aoudad_qa_studies_family(seed: int = _SEED + 1):
    """aoudad_qa_studies: synthetic correctness bench."""
    return _finite_blob(aoudad_qa_studies.bench_aoudad_qa_studies(seed))


def bench_dromedary_qa_studies_family(seed: int = _SEED + 2):
    """dromedary_qa_studies: synthetic correctness bench."""
    return _finite_blob(dromedary_qa_studies.bench_dromedary_qa_studies(seed))


def bench_guanaco_qa_studies_family(seed: int = _SEED + 3):
    """guanaco_qa_studies: synthetic correctness bench."""
    return _finite_blob(guanaco_qa_studies.bench_guanaco_qa_studies(seed))


def bench_salt_qa_studies_family(seed: int = _SEED + 4):
    """salt_qa_studies: synthetic correctness bench."""
    return _finite_blob(salt_qa_studies.bench_salt_qa_studies(seed))


def bench_vicuna_qa_studies_family(seed: int = _SEED + 5):
    """vicuna_qa_studies: synthetic correctness bench."""
    return _finite_blob(vicuna_qa_studies.bench_vicuna_qa_studies(seed))
