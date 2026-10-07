"""Wave-1540 bench adapters: egret canon (SYNTHETIC only)."""

from quant_fund.models import (
    cattle_egret_qa_studies,
    glossy_ibis_qa_studies,
    great_egret_qa_studies,
    sacred_ibis_qa_studies,
    snowy_egret_qa_studies,
    squacco_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15400


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


def bench_cattle_egret_qa_studies_family(seed: int = _SEED + 0):
    """cattle_egret_qa_studies: synthetic correctness bench."""
    return _finite_blob(cattle_egret_qa_studies.bench_cattle_egret_qa_studies(seed))


def bench_glossy_ibis_qa_studies_family(seed: int = _SEED + 1):
    """glossy_ibis_qa_studies: synthetic correctness bench."""
    return _finite_blob(glossy_ibis_qa_studies.bench_glossy_ibis_qa_studies(seed))


def bench_great_egret_qa_studies_family(seed: int = _SEED + 2):
    """great_egret_qa_studies: synthetic correctness bench."""
    return _finite_blob(great_egret_qa_studies.bench_great_egret_qa_studies(seed))


def bench_sacred_ibis_qa_studies_family(seed: int = _SEED + 3):
    """sacred_ibis_qa_studies: synthetic correctness bench."""
    return _finite_blob(sacred_ibis_qa_studies.bench_sacred_ibis_qa_studies(seed))


def bench_snowy_egret_qa_studies_family(seed: int = _SEED + 4):
    """snowy_egret_qa_studies: synthetic correctness bench."""
    return _finite_blob(snowy_egret_qa_studies.bench_snowy_egret_qa_studies(seed))


def bench_squacco_qa_studies_family(seed: int = _SEED + 5):
    """squacco_qa_studies: synthetic correctness bench."""
    return _finite_blob(squacco_qa_studies.bench_squacco_qa_studies(seed))
