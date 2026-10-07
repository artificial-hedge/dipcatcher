"""Wave-1557 bench adapters: amazon-fish canon (SYNTHETIC only)."""

from quant_fund.models import (
    arapaima_qa_studies,
    electric_eel_qa_studies,
    knifefish_qa_studies,
    oscar_qa_studies,
    pacu_qa_studies,
    tetra_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15570


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


def bench_arapaima_qa_studies_family(seed: int = _SEED + 0):
    """arapaima_qa_studies: synthetic correctness bench."""
    return _finite_blob(arapaima_qa_studies.bench_arapaima_qa_studies(seed))


def bench_electric_eel_qa_studies_family(seed: int = _SEED + 1):
    """electric_eel_qa_studies: synthetic correctness bench."""
    return _finite_blob(electric_eel_qa_studies.bench_electric_eel_qa_studies(seed))


def bench_knifefish_qa_studies_family(seed: int = _SEED + 2):
    """knifefish_qa_studies: synthetic correctness bench."""
    return _finite_blob(knifefish_qa_studies.bench_knifefish_qa_studies(seed))


def bench_oscar_qa_studies_family(seed: int = _SEED + 3):
    """oscar_qa_studies: synthetic correctness bench."""
    return _finite_blob(oscar_qa_studies.bench_oscar_qa_studies(seed))


def bench_pacu_qa_studies_family(seed: int = _SEED + 4):
    """pacu_qa_studies: synthetic correctness bench."""
    return _finite_blob(pacu_qa_studies.bench_pacu_qa_studies(seed))


def bench_tetra_qa_studies_family(seed: int = _SEED + 5):
    """tetra_qa_studies: synthetic correctness bench."""
    return _finite_blob(tetra_qa_studies.bench_tetra_qa_studies(seed))
