"""Wave-1267 bench adapters: neuro-symbolic canon (SYNTHETIC only)."""

from quant_fund.models import (
    alpha_tensor_studies,
    differentiable_sat_studies,
    neural_theorem_studies,
    program_synthesis_studies,
    sketch_programming_studies,
    symbolic_regression_dl_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12670


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alpha_tensor_studies_family(seed: int = _SEED + 0):
    """alpha_tensor_studies: synthetic correctness bench."""
    return _finite_blob(alpha_tensor_studies.bench_alpha_tensor_studies(seed))


def bench_differentiable_sat_studies_family(seed: int = _SEED + 1):
    """differentiable_sat_studies: synthetic correctness bench."""
    return _finite_blob(differentiable_sat_studies.bench_differentiable_sat_studies(seed))


def bench_neural_theorem_studies_family(seed: int = _SEED + 2):
    """neural_theorem_studies: synthetic correctness bench."""
    return _finite_blob(neural_theorem_studies.bench_neural_theorem_studies(seed))


def bench_program_synthesis_studies_family(seed: int = _SEED + 3):
    """program_synthesis_studies: synthetic correctness bench."""
    return _finite_blob(program_synthesis_studies.bench_program_synthesis_studies(seed))


def bench_sketch_programming_studies_family(seed: int = _SEED + 4):
    """sketch_programming_studies: synthetic correctness bench."""
    return _finite_blob(sketch_programming_studies.bench_sketch_programming_studies(seed))


def bench_symbolic_regression_dl_studies_family(seed: int = _SEED + 5):
    """symbolic_regression_dl_studies: synthetic correctness bench."""
    return _finite_blob(symbolic_regression_dl_studies.bench_symbolic_regression_dl_studies(seed))
