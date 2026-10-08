"""Wave-1306 bench adapters: GLUE-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    glue_studies,
    mnli_studies,
    qnli_studies,
    rte_studies,
    super_glue_studies,
    wnli_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13060


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


def bench_glue_studies_family(seed: int = _SEED + 0):
    """glue_studies: synthetic correctness bench."""
    return _finite_blob(glue_studies.bench_glue_studies(seed))


def bench_mnli_studies_family(seed: int = _SEED + 1):
    """mnli_studies: synthetic correctness bench."""
    return _finite_blob(mnli_studies.bench_mnli_studies(seed))


def bench_qnli_studies_family(seed: int = _SEED + 2):
    """qnli_studies: synthetic correctness bench."""
    return _finite_blob(qnli_studies.bench_qnli_studies(seed))


def bench_rte_studies_family(seed: int = _SEED + 3):
    """rte_studies: synthetic correctness bench."""
    return _finite_blob(rte_studies.bench_rte_studies(seed))


def bench_super_glue_studies_family(seed: int = _SEED + 4):
    """super_glue_studies: synthetic correctness bench."""
    return _finite_blob(super_glue_studies.bench_super_glue_studies(seed))


def bench_wnli_studies_family(seed: int = _SEED + 5):
    """wnli_studies: synthetic correctness bench."""
    return _finite_blob(wnli_studies.bench_wnli_studies(seed))
