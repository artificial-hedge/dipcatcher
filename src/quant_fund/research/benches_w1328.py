"""Wave-1328 bench adapters: code-eval-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    apps_bench_studies,
    class_eval_studies,
    code_contests_studies,
    multipl_e_studies,
    polyglot_bench_studies,
    repobench_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13280


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


def bench_apps_bench_studies_family(seed: int = _SEED + 0):
    """apps_bench_studies: synthetic correctness bench."""
    return _finite_blob(apps_bench_studies.bench_apps_bench_studies(seed))


def bench_class_eval_studies_family(seed: int = _SEED + 1):
    """class_eval_studies: synthetic correctness bench."""
    return _finite_blob(class_eval_studies.bench_class_eval_studies(seed))


def bench_code_contests_studies_family(seed: int = _SEED + 2):
    """code_contests_studies: synthetic correctness bench."""
    return _finite_blob(code_contests_studies.bench_code_contests_studies(seed))


def bench_multipl_e_studies_family(seed: int = _SEED + 3):
    """multipl_e_studies: synthetic correctness bench."""
    return _finite_blob(multipl_e_studies.bench_multipl_e_studies(seed))


def bench_polyglot_bench_studies_family(seed: int = _SEED + 4):
    """polyglot_bench_studies: synthetic correctness bench."""
    return _finite_blob(polyglot_bench_studies.bench_polyglot_bench_studies(seed))


def bench_repobench_studies_family(seed: int = _SEED + 5):
    """repobench_studies: synthetic correctness bench."""
    return _finite_blob(repobench_studies.bench_repobench_studies(seed))
