"""Wave-1681 bench adapters: african-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    anansi_qa_studies,
    impundulu_qa_studies,
    kalulu_qa_studies,
    mamiwata_qa_studies,
    sasabonsam_qa_studies,
    tokoloshe_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16810


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


def bench_anansi_qa_studies_family(seed: int = _SEED + 0):
    """anansi_qa_studies: synthetic correctness bench."""
    return _finite_blob(anansi_qa_studies.bench_anansi_qa_studies(seed))


def bench_impundulu_qa_studies_family(seed: int = _SEED + 1):
    """impundulu_qa_studies: synthetic correctness bench."""
    return _finite_blob(impundulu_qa_studies.bench_impundulu_qa_studies(seed))


def bench_kalulu_qa_studies_family(seed: int = _SEED + 2):
    """kalulu_qa_studies: synthetic correctness bench."""
    return _finite_blob(kalulu_qa_studies.bench_kalulu_qa_studies(seed))


def bench_mamiwata_qa_studies_family(seed: int = _SEED + 3):
    """mamiwata_qa_studies: synthetic correctness bench."""
    return _finite_blob(mamiwata_qa_studies.bench_mamiwata_qa_studies(seed))


def bench_sasabonsam_qa_studies_family(seed: int = _SEED + 4):
    """sasabonsam_qa_studies: synthetic correctness bench."""
    return _finite_blob(sasabonsam_qa_studies.bench_sasabonsam_qa_studies(seed))


def bench_tokoloshe_qa_studies_family(seed: int = _SEED + 5):
    """tokoloshe_qa_studies: synthetic correctness bench."""
    return _finite_blob(tokoloshe_qa_studies.bench_tokoloshe_qa_studies(seed))
