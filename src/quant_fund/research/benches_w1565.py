"""Wave-1565 bench adapters: annelid canon (SYNTHETIC only)."""

from quant_fund.models import (
    earthworm_qa_studies,
    feather_duster_qa_studies,
    leech_qa_studies,
    lugworm_qa_studies,
    polychaete_qa_studies,
    ragworm_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15650


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


def bench_earthworm_qa_studies_family(seed: int = _SEED + 0):
    """earthworm_qa_studies: synthetic correctness bench."""
    return _finite_blob(earthworm_qa_studies.bench_earthworm_qa_studies(seed))


def bench_feather_duster_qa_studies_family(seed: int = _SEED + 1):
    """feather_duster_qa_studies: synthetic correctness bench."""
    return _finite_blob(feather_duster_qa_studies.bench_feather_duster_qa_studies(seed))


def bench_leech_qa_studies_family(seed: int = _SEED + 2):
    """leech_qa_studies: synthetic correctness bench."""
    return _finite_blob(leech_qa_studies.bench_leech_qa_studies(seed))


def bench_lugworm_qa_studies_family(seed: int = _SEED + 3):
    """lugworm_qa_studies: synthetic correctness bench."""
    return _finite_blob(lugworm_qa_studies.bench_lugworm_qa_studies(seed))


def bench_polychaete_qa_studies_family(seed: int = _SEED + 4):
    """polychaete_qa_studies: synthetic correctness bench."""
    return _finite_blob(polychaete_qa_studies.bench_polychaete_qa_studies(seed))


def bench_ragworm_qa_studies_family(seed: int = _SEED + 5):
    """ragworm_qa_studies: synthetic correctness bench."""
    return _finite_blob(ragworm_qa_studies.bench_ragworm_qa_studies(seed))
