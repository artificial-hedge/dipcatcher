"""Wave-1335 bench adapters: code-eval-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    code_rag_studies,
    codegen_universal_studies,
    long_code_bench_studies,
    odex_eval_studies,
    swe_dev_studies,
    swe_multimodal_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13350


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


def bench_code_rag_studies_family(seed: int = _SEED + 0):
    """code_rag_studies: synthetic correctness bench."""
    return _finite_blob(code_rag_studies.bench_code_rag_studies(seed))


def bench_codegen_universal_studies_family(seed: int = _SEED + 1):
    """codegen_universal_studies: synthetic correctness bench."""
    return _finite_blob(codegen_universal_studies.bench_codegen_universal_studies(seed))


def bench_long_code_bench_studies_family(seed: int = _SEED + 2):
    """long_code_bench_studies: synthetic correctness bench."""
    return _finite_blob(long_code_bench_studies.bench_long_code_bench_studies(seed))


def bench_odex_eval_studies_family(seed: int = _SEED + 3):
    """odex_eval_studies: synthetic correctness bench."""
    return _finite_blob(odex_eval_studies.bench_odex_eval_studies(seed))


def bench_swe_dev_studies_family(seed: int = _SEED + 4):
    """swe_dev_studies: synthetic correctness bench."""
    return _finite_blob(swe_dev_studies.bench_swe_dev_studies(seed))


def bench_swe_multimodal_studies_family(seed: int = _SEED + 5):
    """swe_multimodal_studies: synthetic correctness bench."""
    return _finite_blob(swe_multimodal_studies.bench_swe_multimodal_studies(seed))
