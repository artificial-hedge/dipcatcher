"""Wave-1518 bench adapters: hummingbird-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    coquette_qa_studies,
    fairy_qa_studies,
    jacobin_qa_studies,
    lancebill_qa_studies,
    sabrewing_qa_studies,
    sheartail_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15180


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


def bench_coquette_qa_studies_family(seed: int = _SEED + 0):
    """coquette_qa_studies: synthetic correctness bench."""
    return _finite_blob(coquette_qa_studies.bench_coquette_qa_studies(seed))


def bench_fairy_qa_studies_family(seed: int = _SEED + 1):
    """fairy_qa_studies: synthetic correctness bench."""
    return _finite_blob(fairy_qa_studies.bench_fairy_qa_studies(seed))


def bench_jacobin_qa_studies_family(seed: int = _SEED + 2):
    """jacobin_qa_studies: synthetic correctness bench."""
    return _finite_blob(jacobin_qa_studies.bench_jacobin_qa_studies(seed))


def bench_lancebill_qa_studies_family(seed: int = _SEED + 3):
    """lancebill_qa_studies: synthetic correctness bench."""
    return _finite_blob(lancebill_qa_studies.bench_lancebill_qa_studies(seed))


def bench_sabrewing_qa_studies_family(seed: int = _SEED + 4):
    """sabrewing_qa_studies: synthetic correctness bench."""
    return _finite_blob(sabrewing_qa_studies.bench_sabrewing_qa_studies(seed))


def bench_sheartail_qa_studies_family(seed: int = _SEED + 5):
    """sheartail_qa_studies: synthetic correctness bench."""
    return _finite_blob(sheartail_qa_studies.bench_sheartail_qa_studies(seed))
