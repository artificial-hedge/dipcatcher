"""Wave-1655 bench adapters: heraldic-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    basiliskcock_qa_studies,
    calygreyhound_qa_studies,
    cocatrix_qa_studies,
    gryps_qa_studies,
    mantygre_qa_studies,
    opinicus_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16550


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


def bench_basiliskcock_qa_studies_family(seed: int = _SEED + 0):
    """basiliskcock_qa_studies: synthetic correctness bench."""
    return _finite_blob(basiliskcock_qa_studies.bench_basiliskcock_qa_studies(seed))


def bench_calygreyhound_qa_studies_family(seed: int = _SEED + 1):
    """calygreyhound_qa_studies: synthetic correctness bench."""
    return _finite_blob(calygreyhound_qa_studies.bench_calygreyhound_qa_studies(seed))


def bench_cocatrix_qa_studies_family(seed: int = _SEED + 2):
    """cocatrix_qa_studies: synthetic correctness bench."""
    return _finite_blob(cocatrix_qa_studies.bench_cocatrix_qa_studies(seed))


def bench_gryps_qa_studies_family(seed: int = _SEED + 3):
    """gryps_qa_studies: synthetic correctness bench."""
    return _finite_blob(gryps_qa_studies.bench_gryps_qa_studies(seed))


def bench_mantygre_qa_studies_family(seed: int = _SEED + 4):
    """mantygre_qa_studies: synthetic correctness bench."""
    return _finite_blob(mantygre_qa_studies.bench_mantygre_qa_studies(seed))


def bench_opinicus_qa_studies_family(seed: int = _SEED + 5):
    """opinicus_qa_studies: synthetic correctness bench."""
    return _finite_blob(opinicus_qa_studies.bench_opinicus_qa_studies(seed))
