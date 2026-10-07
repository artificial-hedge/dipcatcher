"""Wave-1303 bench adapters: LLM-academic-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    bbh_studies,
    gsm8k_studies,
    humaneval_studies,
    ifeval_studies,
    mmlu_studies,
    mt_bench_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13030


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


def bench_bbh_studies_family(seed: int = _SEED + 0):
    """bbh_studies: synthetic correctness bench."""
    return _finite_blob(bbh_studies.bench_bbh_studies(seed))


def bench_gsm8k_studies_family(seed: int = _SEED + 1):
    """gsm8k_studies: synthetic correctness bench."""
    return _finite_blob(gsm8k_studies.bench_gsm8k_studies(seed))


def bench_humaneval_studies_family(seed: int = _SEED + 2):
    """humaneval_studies: synthetic correctness bench."""
    return _finite_blob(humaneval_studies.bench_humaneval_studies(seed))


def bench_ifeval_studies_family(seed: int = _SEED + 3):
    """ifeval_studies: synthetic correctness bench."""
    return _finite_blob(ifeval_studies.bench_ifeval_studies(seed))


def bench_mmlu_studies_family(seed: int = _SEED + 4):
    """mmlu_studies: synthetic correctness bench."""
    return _finite_blob(mmlu_studies.bench_mmlu_studies(seed))


def bench_mt_bench_studies_family(seed: int = _SEED + 5):
    """mt_bench_studies: synthetic correctness bench."""
    return _finite_blob(mt_bench_studies.bench_mt_bench_studies(seed))
