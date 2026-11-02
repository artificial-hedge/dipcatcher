"""Shrunk scorecard budgets for the pytest lane.

``dipcatcher research`` keeps each bench's default fixture. Pytest sets
``DIP_SCORECARD_BUDGET=ci`` (see ``tests/conftest.py``) so the scorecard
adapters use smaller fixtures. Model unit tests that need a directional
margin call the benches directly and are not affected.
Explicit keyword arguments from the caller win over the CI map.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from typing import Any

# Budgets copied from the model unit tests that already require a finite blob
# (and, where those tests check one, the same directional claim).
CI_BENCH_BUDGETS: dict[str, dict[str, int]] = {
    "bench_cql_agent": {"n_ep": 60, "iters": 200},
    "bench_iql_agent": {"n_ep": 60, "iters": 200},
    "bench_neural_process": {"n_train": 60, "n_test": 10, "iters": 60},
    # Smaller than the model-unit fixtures. Those tests call the benches
    # directly and still require a positive margin. The scorecard only
    # requires a finite blob, which these seeds still emit.
    "bench_decision_transformer": {"n_ep": 30, "iters": 60},
    "bench_trajectory_transformer": {"n_ep": 30, "iters": 40},
    "bench_sac_agent": {"steps": 800, "batch": 64},
    "bench_gail_imitation": {"n_ep": 40, "iters": 200},
    "bench_deep_kernel_gp": {"n_train": 40, "n_test": 8, "iters": 30},
    "bench_attentive_np": {"n_train": 60, "n_test": 10, "iters": 60},
    "bench_convnp": {"n_train": 60, "n_test": 10, "iters": 60},
    "bench_mambats": {"n": 120, "iters": 200},
    "bench_reformer_lsh": {"n_train": 80, "n_test": 40, "m_pairs": 16, "iters": 60},
    "bench_memorizing_transformer": {"n_train": 80, "n_test": 40, "m_pairs": 16, "iters": 60},
    "bench_ntm_memory": {"n_train": 60, "n_test": 30, "t": 5, "iters": 40},
    "bench_dnc_memory": {"n_train": 60, "n_test": 30, "t": 5, "iters": 40},
    "bench_vrnn_seq": {"steps": 100, "T": 8},
    "bench_redq": {"steps": 400, "N": 4},
    "bench_listmle_ltr": {"iters": 10},
    "bench_colbert_late": {"n_docs": 80, "n_queries": 30, "doc_len": 20, "iters": 60},
    "bench_input_convex": {"n_train": 400, "n_test": 80, "iters": 60},
    "bench_diff_mpc": {"n_train": 8, "n_test": 4, "iters": 15, "t_final": 10},
    "bench_itransformer": {"n": 120, "iters": 200},
    "bench_tcn_forecaster": {"n": 120, "iters": 200},
    "bench_ft_transformer": {"n": 120, "iters": 200},
    "bench_crossformer": {"n": 120, "iters": 200},
    "bench_informer_attn": {"n": 480, "steps": 40, "batch": 32},
    # Closed forms stay exact at n=400; the Sinkhorn gap stays above 10x.
    "bench_optimal_transport": {"n": 400},
    # Left-tail and Hawkes-Fano inequalities in test_benches_w20 still hold.
    "arl_mm_bench": {"n_rounds": 2, "n_eval_episodes": 8, "n_boot": 80},
}


def ci_scorecard() -> bool:
    return os.environ.get("DIP_SCORECARD_BUDGET", "full") == "ci"


# Seeded live-optional benches do not read the research panel. A second
# run_research in the same process (the fingerprint test on one xdist
# worker) reuses the blob instead of rebuilding it.
_CI_OPTIONAL_BLOBS: dict[str, dict[str, Any]] = {}


def cached_optional_blobs(build: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Return ``build()``, cached while the CI scorecard budget is on."""
    if ci_scorecard():
        cached = _CI_OPTIONAL_BLOBS.get("ci")
        if cached is not None:
            return copy.deepcopy(cached)
    blobs = build()
    if ci_scorecard():
        _CI_OPTIONAL_BLOBS["ci"] = copy.deepcopy(blobs)
    return blobs


def call_bench(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with CI budget kwargs merged under the caller's kwargs."""
    if ci_scorecard():
        extra = CI_BENCH_BUDGETS.get(getattr(fn, "__name__", ""), {})
        if extra:
            kwargs = {**extra, **kwargs}
    return fn(*args, **kwargs)
