"""Wave-1477 bench adapters: seabird canon (SYNTHETIC only)."""

from quant_fund.models import (
    albatross_qa_studies,
    gannet_qa_studies,
    petrel_qa_studies,
    puffin_qa_studies,
    shearwater_qa_studies,
    skua_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14770


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


def bench_albatross_qa_studies_family(seed: int = _SEED + 0):
    """albatross_qa_studies: synthetic correctness bench."""
    return _finite_blob(albatross_qa_studies.bench_albatross_qa_studies(seed))


def bench_gannet_qa_studies_family(seed: int = _SEED + 1):
    """gannet_qa_studies: synthetic correctness bench."""
    return _finite_blob(gannet_qa_studies.bench_gannet_qa_studies(seed))


def bench_petrel_qa_studies_family(seed: int = _SEED + 2):
    """petrel_qa_studies: synthetic correctness bench."""
    return _finite_blob(petrel_qa_studies.bench_petrel_qa_studies(seed))


def bench_puffin_qa_studies_family(seed: int = _SEED + 3):
    """puffin_qa_studies: synthetic correctness bench."""
    return _finite_blob(puffin_qa_studies.bench_puffin_qa_studies(seed))


def bench_shearwater_qa_studies_family(seed: int = _SEED + 4):
    """shearwater_qa_studies: synthetic correctness bench."""
    return _finite_blob(shearwater_qa_studies.bench_shearwater_qa_studies(seed))


def bench_skua_qa_studies_family(seed: int = _SEED + 5):
    """skua_qa_studies: synthetic correctness bench."""
    return _finite_blob(skua_qa_studies.bench_skua_qa_studies(seed))
