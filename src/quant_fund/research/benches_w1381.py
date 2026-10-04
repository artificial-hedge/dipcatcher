"""Wave-1381 bench adapters: dialogue-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    begins_lite_studies,
    diamonds_lite_studies,
    faithful_dial_studies,
    multi_woz_studies,
    top_dialog_studies,
    wow_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13810


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_begins_lite_studies_family(seed: int = _SEED + 0):
    """begins_lite_studies: synthetic correctness bench."""
    return _finite_blob(begins_lite_studies.bench_begins_lite_studies(seed))


def bench_diamonds_lite_studies_family(seed: int = _SEED + 1):
    """diamonds_lite_studies: synthetic correctness bench."""
    return _finite_blob(diamonds_lite_studies.bench_diamonds_lite_studies(seed))


def bench_faithful_dial_studies_family(seed: int = _SEED + 2):
    """faithful_dial_studies: synthetic correctness bench."""
    return _finite_blob(faithful_dial_studies.bench_faithful_dial_studies(seed))


def bench_multi_woz_studies_family(seed: int = _SEED + 3):
    """multi_woz_studies: synthetic correctness bench."""
    return _finite_blob(multi_woz_studies.bench_multi_woz_studies(seed))


def bench_top_dialog_studies_family(seed: int = _SEED + 4):
    """top_dialog_studies: synthetic correctness bench."""
    return _finite_blob(top_dialog_studies.bench_top_dialog_studies(seed))


def bench_wow_lite_studies_family(seed: int = _SEED + 5):
    """wow_lite_studies: synthetic correctness bench."""
    return _finite_blob(wow_lite_studies.bench_wow_lite_studies(seed))
