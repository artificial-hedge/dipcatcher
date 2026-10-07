"""Wave-1465 bench adapters: forge canon (SYNTHETIC only)."""

from quant_fund.models import (
    citadel_qa_studies,
    forge_qa_studies,
    grotto_qa_studies,
    lighthouse_qa_studies,
    quarry_qa_studies,
    vault_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14650


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


def bench_citadel_qa_studies_family(seed: int = _SEED + 0):
    """citadel_qa_studies: synthetic correctness bench."""
    return _finite_blob(citadel_qa_studies.bench_citadel_qa_studies(seed))


def bench_forge_qa_studies_family(seed: int = _SEED + 1):
    """forge_qa_studies: synthetic correctness bench."""
    return _finite_blob(forge_qa_studies.bench_forge_qa_studies(seed))


def bench_grotto_qa_studies_family(seed: int = _SEED + 2):
    """grotto_qa_studies: synthetic correctness bench."""
    return _finite_blob(grotto_qa_studies.bench_grotto_qa_studies(seed))


def bench_lighthouse_qa_studies_family(seed: int = _SEED + 3):
    """lighthouse_qa_studies: synthetic correctness bench."""
    return _finite_blob(lighthouse_qa_studies.bench_lighthouse_qa_studies(seed))


def bench_quarry_qa_studies_family(seed: int = _SEED + 4):
    """quarry_qa_studies: synthetic correctness bench."""
    return _finite_blob(quarry_qa_studies.bench_quarry_qa_studies(seed))


def bench_vault_qa_studies_family(seed: int = _SEED + 5):
    """vault_qa_studies: synthetic correctness bench."""
    return _finite_blob(vault_qa_studies.bench_vault_qa_studies(seed))
