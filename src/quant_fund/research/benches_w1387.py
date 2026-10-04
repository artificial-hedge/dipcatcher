"""Wave-1387 bench adapters: tool-use canon (SYNTHETIC only)."""

from quant_fund.models import (
    api_blend_studies,
    bfcl_v3_studies,
    gorilla_eval_studies,
    gta_bench_studies,
    seal_tools_studies,
    stabletoolbench_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13870


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_api_blend_studies_family(seed: int = _SEED + 0):
    """api_blend_studies: synthetic correctness bench."""
    return _finite_blob(api_blend_studies.bench_api_blend_studies(seed))


def bench_bfcl_v3_studies_family(seed: int = _SEED + 1):
    """bfcl_v3_studies: synthetic correctness bench."""
    return _finite_blob(bfcl_v3_studies.bench_bfcl_v3_studies(seed))


def bench_gorilla_eval_studies_family(seed: int = _SEED + 2):
    """gorilla_eval_studies: synthetic correctness bench."""
    return _finite_blob(gorilla_eval_studies.bench_gorilla_eval_studies(seed))


def bench_gta_bench_studies_family(seed: int = _SEED + 3):
    """gta_bench_studies: synthetic correctness bench."""
    return _finite_blob(gta_bench_studies.bench_gta_bench_studies(seed))


def bench_seal_tools_studies_family(seed: int = _SEED + 4):
    """seal_tools_studies: synthetic correctness bench."""
    return _finite_blob(seal_tools_studies.bench_seal_tools_studies(seed))


def bench_stabletoolbench_studies_family(seed: int = _SEED + 5):
    """stabletoolbench_studies: synthetic correctness bench."""
    return _finite_blob(stabletoolbench_studies.bench_stabletoolbench_studies(seed))
