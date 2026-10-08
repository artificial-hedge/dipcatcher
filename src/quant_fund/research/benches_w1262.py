"""Wave-1262 bench adapters: target-trial/RWE canon (SYNTHETIC only)."""

from quant_fund.models import (
    dynamic_borrowing_studies,
    e_value_studies,
    master_protocol_studies,
    stepped_wedge_studies,
    target_trial_emulation_studies,
    win_ratio_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12620


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


def bench_dynamic_borrowing_studies_family(seed: int = _SEED + 0):
    """dynamic_borrowing_studies: synthetic correctness bench."""
    return _finite_blob(dynamic_borrowing_studies.bench_dynamic_borrowing_studies(seed))


def bench_e_value_studies_family(seed: int = _SEED + 1):
    """e_value_studies: synthetic correctness bench."""
    return _finite_blob(e_value_studies.bench_e_value_studies(seed))


def bench_master_protocol_studies_family(seed: int = _SEED + 2):
    """master_protocol_studies: synthetic correctness bench."""
    return _finite_blob(master_protocol_studies.bench_master_protocol_studies(seed))


def bench_stepped_wedge_studies_family(seed: int = _SEED + 3):
    """stepped_wedge_studies: synthetic correctness bench."""
    return _finite_blob(stepped_wedge_studies.bench_stepped_wedge_studies(seed))


def bench_target_trial_emulation_studies_family(seed: int = _SEED + 4):
    """target_trial_emulation_studies: synthetic correctness bench."""
    return _finite_blob(target_trial_emulation_studies.bench_target_trial_emulation_studies(seed))


def bench_win_ratio_studies_family(seed: int = _SEED + 5):
    """win_ratio_studies: synthetic correctness bench."""
    return _finite_blob(win_ratio_studies.bench_win_ratio_studies(seed))
