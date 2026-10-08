"""Wave-1259 bench adapters: drug-discovery canon (SYNTHETIC only)."""

from quant_fund.models import (
    admet_studies,
    de_novo_design_studies,
    docking_studies,
    lead_optimization_studies,
    qsar_studies,
    virtual_screening_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12590


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


def bench_qsar_studies_family(seed: int = _SEED + 0):
    """qsar_studies: synthetic correctness bench."""
    return _finite_blob(qsar_studies.bench_qsar_studies(seed))


def bench_docking_studies_family(seed: int = _SEED + 1):
    """docking_studies: synthetic correctness bench."""
    return _finite_blob(docking_studies.bench_docking_studies(seed))


def bench_admet_studies_family(seed: int = _SEED + 2):
    """admet_studies: synthetic correctness bench."""
    return _finite_blob(admet_studies.bench_admet_studies(seed))


def bench_lead_optimization_studies_family(seed: int = _SEED + 3):
    """lead_optimization_studies: synthetic correctness bench."""
    return _finite_blob(lead_optimization_studies.bench_lead_optimization_studies(seed))


def bench_virtual_screening_studies_family(seed: int = _SEED + 4):
    """virtual_screening_studies: synthetic correctness bench."""
    return _finite_blob(virtual_screening_studies.bench_virtual_screening_studies(seed))


def bench_de_novo_design_studies_family(seed: int = _SEED + 5):
    """de_novo_design_studies: synthetic correctness bench."""
    return _finite_blob(de_novo_design_studies.bench_de_novo_design_studies(seed))
