"""Wave-1364 bench adapters: sentence-pair canon (SYNTHETIC only)."""

from quant_fund.models import (
    anli_lite_studies,
    mnli_lite_studies,
    mrpc_lite_studies,
    paws_lite_studies,
    quora_dup_studies,
    rte_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13640


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


def bench_anli_lite_studies_family(seed: int = _SEED + 0):
    """anli_lite_studies: synthetic correctness bench."""
    return _finite_blob(anli_lite_studies.bench_anli_lite_studies(seed))


def bench_mnli_lite_studies_family(seed: int = _SEED + 1):
    """mnli_lite_studies: synthetic correctness bench."""
    return _finite_blob(mnli_lite_studies.bench_mnli_lite_studies(seed))


def bench_mrpc_lite_studies_family(seed: int = _SEED + 2):
    """mrpc_lite_studies: synthetic correctness bench."""
    return _finite_blob(mrpc_lite_studies.bench_mrpc_lite_studies(seed))


def bench_paws_lite_studies_family(seed: int = _SEED + 3):
    """paws_lite_studies: synthetic correctness bench."""
    return _finite_blob(paws_lite_studies.bench_paws_lite_studies(seed))


def bench_quora_dup_studies_family(seed: int = _SEED + 4):
    """quora_dup_studies: synthetic correctness bench."""
    return _finite_blob(quora_dup_studies.bench_quora_dup_studies(seed))


def bench_rte_lite_studies_family(seed: int = _SEED + 5):
    """rte_lite_studies: synthetic correctness bench."""
    return _finite_blob(rte_lite_studies.bench_rte_lite_studies(seed))
