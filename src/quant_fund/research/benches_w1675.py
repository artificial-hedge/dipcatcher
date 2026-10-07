"""Wave-1675 bench adapters: aztec-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    chaneque_qa_studies,
    cihuateteo_qa_studies,
    nagual_qa_studies,
    tlalocan_qa_studies,
    tzitzimitl_qa_studies,
    xiuhcoatl_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16750


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


def bench_chaneque_qa_studies_family(seed: int = _SEED + 0):
    """chaneque_qa_studies: synthetic correctness bench."""
    return _finite_blob(chaneque_qa_studies.bench_chaneque_qa_studies(seed))


def bench_cihuateteo_qa_studies_family(seed: int = _SEED + 1):
    """cihuateteo_qa_studies: synthetic correctness bench."""
    return _finite_blob(cihuateteo_qa_studies.bench_cihuateteo_qa_studies(seed))


def bench_nagual_qa_studies_family(seed: int = _SEED + 2):
    """nagual_qa_studies: synthetic correctness bench."""
    return _finite_blob(nagual_qa_studies.bench_nagual_qa_studies(seed))


def bench_tlalocan_qa_studies_family(seed: int = _SEED + 3):
    """tlalocan_qa_studies: synthetic correctness bench."""
    return _finite_blob(tlalocan_qa_studies.bench_tlalocan_qa_studies(seed))


def bench_tzitzimitl_qa_studies_family(seed: int = _SEED + 4):
    """tzitzimitl_qa_studies: synthetic correctness bench."""
    return _finite_blob(tzitzimitl_qa_studies.bench_tzitzimitl_qa_studies(seed))


def bench_xiuhcoatl_qa_studies_family(seed: int = _SEED + 5):
    """xiuhcoatl_qa_studies: synthetic correctness bench."""
    return _finite_blob(xiuhcoatl_qa_studies.bench_xiuhcoatl_qa_studies(seed))
