"""Wave-1281 bench adapters: reasoning-prompt canon (SYNTHETIC only)."""

from quant_fund.models import (
    analogical_prompting_studies,
    graph_of_thought_studies,
    least_to_most_studies,
    plan_and_solve_studies,
    step_back_studies,
    tree_of_thought_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12810


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_analogical_prompting_studies_family(seed: int = _SEED + 0):
    """analogical_prompting_studies: synthetic correctness bench."""
    return _finite_blob(analogical_prompting_studies.bench_analogical_prompting_studies(seed))


def bench_graph_of_thought_studies_family(seed: int = _SEED + 1):
    """graph_of_thought_studies: synthetic correctness bench."""
    return _finite_blob(graph_of_thought_studies.bench_graph_of_thought_studies(seed))


def bench_least_to_most_studies_family(seed: int = _SEED + 2):
    """least_to_most_studies: synthetic correctness bench."""
    return _finite_blob(least_to_most_studies.bench_least_to_most_studies(seed))


def bench_plan_and_solve_studies_family(seed: int = _SEED + 3):
    """plan_and_solve_studies: synthetic correctness bench."""
    return _finite_blob(plan_and_solve_studies.bench_plan_and_solve_studies(seed))


def bench_step_back_studies_family(seed: int = _SEED + 4):
    """step_back_studies: synthetic correctness bench."""
    return _finite_blob(step_back_studies.bench_step_back_studies(seed))


def bench_tree_of_thought_studies_family(seed: int = _SEED + 5):
    """tree_of_thought_studies: synthetic correctness bench."""
    return _finite_blob(tree_of_thought_studies.bench_tree_of_thought_studies(seed))
