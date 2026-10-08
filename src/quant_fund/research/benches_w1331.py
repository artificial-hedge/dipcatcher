"""Wave-1331 bench adapters: code-eval-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    codescope_studies,
    concode_eval_studies,
    crosscodeeval_studies,
    mer_bench_studies,
    project_eval_studies,
    swe_bench_verified_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13310


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


def bench_codescope_studies_family(seed: int = _SEED + 0):
    """codescope_studies: synthetic correctness bench."""
    return _finite_blob(codescope_studies.bench_codescope_studies(seed))


def bench_concode_eval_studies_family(seed: int = _SEED + 1):
    """concode_eval_studies: synthetic correctness bench."""
    return _finite_blob(concode_eval_studies.bench_concode_eval_studies(seed))


def bench_crosscodeeval_studies_family(seed: int = _SEED + 2):
    """crosscodeeval_studies: synthetic correctness bench."""
    return _finite_blob(crosscodeeval_studies.bench_crosscodeeval_studies(seed))


def bench_mer_bench_studies_family(seed: int = _SEED + 3):
    """mer_bench_studies: synthetic correctness bench."""
    return _finite_blob(mer_bench_studies.bench_mer_bench_studies(seed))


def bench_project_eval_studies_family(seed: int = _SEED + 4):
    """project_eval_studies: synthetic correctness bench."""
    return _finite_blob(project_eval_studies.bench_project_eval_studies(seed))


def bench_swe_bench_verified_studies_family(seed: int = _SEED + 5):
    """swe_bench_verified_studies: synthetic correctness bench."""
    return _finite_blob(swe_bench_verified_studies.bench_swe_bench_verified_studies(seed))
