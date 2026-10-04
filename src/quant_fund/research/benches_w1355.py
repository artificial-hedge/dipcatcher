"""Wave-1355 bench adapters: commonsense-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    arc_hard2_studies,
    csqa_lite_studies,
    hellaswag_lite_studies,
    piqa_lite_studies,
    prost_lite_studies,
    swag_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13550


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_arc_hard2_studies_family(seed: int = _SEED + 0):
    """arc_hard2_studies: synthetic correctness bench."""
    return _finite_blob(arc_hard2_studies.bench_arc_hard2_studies(seed))


def bench_csqa_lite_studies_family(seed: int = _SEED + 1):
    """csqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(csqa_lite_studies.bench_csqa_lite_studies(seed))


def bench_hellaswag_lite_studies_family(seed: int = _SEED + 2):
    """hellaswag_lite_studies: synthetic correctness bench."""
    return _finite_blob(hellaswag_lite_studies.bench_hellaswag_lite_studies(seed))


def bench_piqa_lite_studies_family(seed: int = _SEED + 3):
    """piqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(piqa_lite_studies.bench_piqa_lite_studies(seed))


def bench_prost_lite_studies_family(seed: int = _SEED + 4):
    """prost_lite_studies: synthetic correctness bench."""
    return _finite_blob(prost_lite_studies.bench_prost_lite_studies(seed))


def bench_swag_lite_studies_family(seed: int = _SEED + 5):
    """swag_lite_studies: synthetic correctness bench."""
    return _finite_blob(swag_lite_studies.bench_swag_lite_studies(seed))
