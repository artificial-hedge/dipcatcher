"""Wave-1644 bench adapters: gorgon canon (SYNTHETIC only)."""

from quant_fund.models import (
    charybdis_qa_studies,
    cyclops_2_qa_studies,
    hydra_3_qa_studies,
    medusa_2_qa_studies,
    scylla_qa_studies,
    siren_2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16440


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


def bench_charybdis_qa_studies_family(seed: int = _SEED + 0):
    """charybdis_qa_studies: synthetic correctness bench."""
    return _finite_blob(charybdis_qa_studies.bench_charybdis_qa_studies(seed))


def bench_cyclops_2_qa_studies_family(seed: int = _SEED + 1):
    """cyclops_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(cyclops_2_qa_studies.bench_cyclops_2_qa_studies(seed))


def bench_hydra_3_qa_studies_family(seed: int = _SEED + 2):
    """hydra_3_qa_studies: synthetic correctness bench."""
    return _finite_blob(hydra_3_qa_studies.bench_hydra_3_qa_studies(seed))


def bench_medusa_2_qa_studies_family(seed: int = _SEED + 3):
    """medusa_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(medusa_2_qa_studies.bench_medusa_2_qa_studies(seed))


def bench_scylla_qa_studies_family(seed: int = _SEED + 4):
    """scylla_qa_studies: synthetic correctness bench."""
    return _finite_blob(scylla_qa_studies.bench_scylla_qa_studies(seed))


def bench_siren_2_qa_studies_family(seed: int = _SEED + 5):
    """siren_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(siren_2_qa_studies.bench_siren_2_qa_studies(seed))
