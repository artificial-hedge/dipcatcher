"""Wave-1613 bench adapters: bovine canon (SYNTHETIC only)."""

from quant_fund.models import (
    aurochs_qa_studies,
    banteng_qa_studies,
    gaur_qa_studies,
    saola_qa_studies,
    tamaraw_qa_studies,
    yak_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16130


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


def bench_aurochs_qa_studies_family(seed: int = _SEED + 0):
    """aurochs_qa_studies: synthetic correctness bench."""
    return _finite_blob(aurochs_qa_studies.bench_aurochs_qa_studies(seed))


def bench_banteng_qa_studies_family(seed: int = _SEED + 1):
    """banteng_qa_studies: synthetic correctness bench."""
    return _finite_blob(banteng_qa_studies.bench_banteng_qa_studies(seed))


def bench_gaur_qa_studies_family(seed: int = _SEED + 2):
    """gaur_qa_studies: synthetic correctness bench."""
    return _finite_blob(gaur_qa_studies.bench_gaur_qa_studies(seed))


def bench_saola_qa_studies_family(seed: int = _SEED + 3):
    """saola_qa_studies: synthetic correctness bench."""
    return _finite_blob(saola_qa_studies.bench_saola_qa_studies(seed))


def bench_tamaraw_qa_studies_family(seed: int = _SEED + 4):
    """tamaraw_qa_studies: synthetic correctness bench."""
    return _finite_blob(tamaraw_qa_studies.bench_tamaraw_qa_studies(seed))


def bench_yak_qa_studies_family(seed: int = _SEED + 5):
    """yak_qa_studies: synthetic correctness bench."""
    return _finite_blob(yak_qa_studies.bench_yak_qa_studies(seed))
