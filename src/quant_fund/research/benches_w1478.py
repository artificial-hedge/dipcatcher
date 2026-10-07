"""Wave-1478 bench adapters: antelope canon (SYNTHETIC only)."""

from quant_fund.models import (
    antelope_qa_studies,
    eland_qa_studies,
    impala_qa_studies,
    kudu_qa_studies,
    oryx_qa_studies,
    springbok_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14780


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


def bench_antelope_qa_studies_family(seed: int = _SEED + 0):
    """antelope_qa_studies: synthetic correctness bench."""
    return _finite_blob(antelope_qa_studies.bench_antelope_qa_studies(seed))


def bench_eland_qa_studies_family(seed: int = _SEED + 1):
    """eland_qa_studies: synthetic correctness bench."""
    return _finite_blob(eland_qa_studies.bench_eland_qa_studies(seed))


def bench_impala_qa_studies_family(seed: int = _SEED + 2):
    """impala_qa_studies: synthetic correctness bench."""
    return _finite_blob(impala_qa_studies.bench_impala_qa_studies(seed))


def bench_kudu_qa_studies_family(seed: int = _SEED + 3):
    """kudu_qa_studies: synthetic correctness bench."""
    return _finite_blob(kudu_qa_studies.bench_kudu_qa_studies(seed))


def bench_oryx_qa_studies_family(seed: int = _SEED + 4):
    """oryx_qa_studies: synthetic correctness bench."""
    return _finite_blob(oryx_qa_studies.bench_oryx_qa_studies(seed))


def bench_springbok_qa_studies_family(seed: int = _SEED + 5):
    """springbok_qa_studies: synthetic correctness bench."""
    return _finite_blob(springbok_qa_studies.bench_springbok_qa_studies(seed))
