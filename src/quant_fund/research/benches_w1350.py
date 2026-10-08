"""Wave-1350 bench adapters: compositional-generalization canon (SYNTHETIC only)."""

from quant_fund.models import (
    dyck_lang_studies,
    hops_add_studies,
    lcmc_lite_studies,
    mco_lite_studies,
    scan_cfsp_studies,
    shuffle_expr_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13500


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


def bench_dyck_lang_studies_family(seed: int = _SEED + 0):
    """dyck_lang_studies: synthetic correctness bench."""
    return _finite_blob(dyck_lang_studies.bench_dyck_lang_studies(seed))


def bench_hops_add_studies_family(seed: int = _SEED + 1):
    """hops_add_studies: synthetic correctness bench."""
    return _finite_blob(hops_add_studies.bench_hops_add_studies(seed))


def bench_lcmc_lite_studies_family(seed: int = _SEED + 2):
    """lcmc_lite_studies: synthetic correctness bench."""
    return _finite_blob(lcmc_lite_studies.bench_lcmc_lite_studies(seed))


def bench_mco_lite_studies_family(seed: int = _SEED + 3):
    """mco_lite_studies: synthetic correctness bench."""
    return _finite_blob(mco_lite_studies.bench_mco_lite_studies(seed))


def bench_scan_cfsp_studies_family(seed: int = _SEED + 4):
    """scan_cfsp_studies: synthetic correctness bench."""
    return _finite_blob(scan_cfsp_studies.bench_scan_cfsp_studies(seed))


def bench_shuffle_expr_studies_family(seed: int = _SEED + 5):
    """shuffle_expr_studies: synthetic correctness bench."""
    return _finite_blob(shuffle_expr_studies.bench_shuffle_expr_studies(seed))
