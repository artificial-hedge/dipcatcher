"""Wave-1455 bench adapters: amphibian canon (SYNTHETIC only)."""

from quant_fund.models import (
    axolotl_qa_studies,
    bullfrog_qa_studies,
    newt_qa_studies,
    salamander_qa_studies,
    toad_qa_studies,
    tree_frog_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14550


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_axolotl_qa_studies_family(seed: int = _SEED + 0):
    """axolotl_qa_studies: synthetic correctness bench."""
    return _finite_blob(axolotl_qa_studies.bench_axolotl_qa_studies(seed))


def bench_bullfrog_qa_studies_family(seed: int = _SEED + 1):
    """bullfrog_qa_studies: synthetic correctness bench."""
    return _finite_blob(bullfrog_qa_studies.bench_bullfrog_qa_studies(seed))


def bench_newt_qa_studies_family(seed: int = _SEED + 2):
    """newt_qa_studies: synthetic correctness bench."""
    return _finite_blob(newt_qa_studies.bench_newt_qa_studies(seed))


def bench_salamander_qa_studies_family(seed: int = _SEED + 3):
    """salamander_qa_studies: synthetic correctness bench."""
    return _finite_blob(salamander_qa_studies.bench_salamander_qa_studies(seed))


def bench_toad_qa_studies_family(seed: int = _SEED + 4):
    """toad_qa_studies: synthetic correctness bench."""
    return _finite_blob(toad_qa_studies.bench_toad_qa_studies(seed))


def bench_tree_frog_qa_studies_family(seed: int = _SEED + 5):
    """tree_frog_qa_studies: synthetic correctness bench."""
    return _finite_blob(tree_frog_qa_studies.bench_tree_frog_qa_studies(seed))
