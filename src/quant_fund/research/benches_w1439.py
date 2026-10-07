"""Wave-1439 bench adapters: landform canon (SYNTHETIC only)."""

from quant_fund.models import (
    cliff_qa_studies,
    crater_qa_studies,
    dune_qa_studies,
    fjord_qa_studies,
    gorge_qa_studies,
    mesa_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14390


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


def bench_cliff_qa_studies_family(seed: int = _SEED + 0):
    """cliff_qa_studies: synthetic correctness bench."""
    return _finite_blob(cliff_qa_studies.bench_cliff_qa_studies(seed))


def bench_crater_qa_studies_family(seed: int = _SEED + 1):
    """crater_qa_studies: synthetic correctness bench."""
    return _finite_blob(crater_qa_studies.bench_crater_qa_studies(seed))


def bench_dune_qa_studies_family(seed: int = _SEED + 2):
    """dune_qa_studies: synthetic correctness bench."""
    return _finite_blob(dune_qa_studies.bench_dune_qa_studies(seed))


def bench_fjord_qa_studies_family(seed: int = _SEED + 3):
    """fjord_qa_studies: synthetic correctness bench."""
    return _finite_blob(fjord_qa_studies.bench_fjord_qa_studies(seed))


def bench_gorge_qa_studies_family(seed: int = _SEED + 4):
    """gorge_qa_studies: synthetic correctness bench."""
    return _finite_blob(gorge_qa_studies.bench_gorge_qa_studies(seed))


def bench_mesa_qa_studies_family(seed: int = _SEED + 5):
    """mesa_qa_studies: synthetic correctness bench."""
    return _finite_blob(mesa_qa_studies.bench_mesa_qa_studies(seed))
