"""Wave-1373 bench adapters: commonsense-reasoning canon (SYNTHETIC only)."""

from quant_fund.models import (
    commonsense_lite_studies,
    logi_qa_studies,
    mr_lite_studies,
    muin_lite_studies,
    qasc_sci2_studies,
    winogrande_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13730


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


def bench_commonsense_lite_studies_family(seed: int = _SEED + 0):
    """commonsense_lite_studies: synthetic correctness bench."""
    return _finite_blob(commonsense_lite_studies.bench_commonsense_lite_studies(seed))


def bench_logi_qa_studies_family(seed: int = _SEED + 1):
    """logi_qa_studies: synthetic correctness bench."""
    return _finite_blob(logi_qa_studies.bench_logi_qa_studies(seed))


def bench_mr_lite_studies_family(seed: int = _SEED + 2):
    """mr_lite_studies: synthetic correctness bench."""
    return _finite_blob(mr_lite_studies.bench_mr_lite_studies(seed))


def bench_muin_lite_studies_family(seed: int = _SEED + 3):
    """muin_lite_studies: synthetic correctness bench."""
    return _finite_blob(muin_lite_studies.bench_muin_lite_studies(seed))


def bench_qasc_sci2_studies_family(seed: int = _SEED + 4):
    """qasc_sci2_studies: synthetic correctness bench."""
    return _finite_blob(qasc_sci2_studies.bench_qasc_sci2_studies(seed))


def bench_winogrande_lite_studies_family(seed: int = _SEED + 5):
    """winogrande_lite_studies: synthetic correctness bench."""
    return _finite_blob(winogrande_lite_studies.bench_winogrande_lite_studies(seed))
