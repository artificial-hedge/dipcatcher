"""Wave-1357 bench adapters: reading-comp-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    hendrycks_test_studies,
    hotpot_lite_studies,
    multirc_lite_studies,
    quoref_lite_studies,
    record_lite_studies,
    squad_lite2_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13570


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_hendrycks_test_studies_family(seed: int = _SEED + 0):
    """hendrycks_test_studies: synthetic correctness bench."""
    return _finite_blob(hendrycks_test_studies.bench_hendrycks_test_studies(seed))


def bench_hotpot_lite_studies_family(seed: int = _SEED + 1):
    """hotpot_lite_studies: synthetic correctness bench."""
    return _finite_blob(hotpot_lite_studies.bench_hotpot_lite_studies(seed))


def bench_multirc_lite_studies_family(seed: int = _SEED + 2):
    """multirc_lite_studies: synthetic correctness bench."""
    return _finite_blob(multirc_lite_studies.bench_multirc_lite_studies(seed))


def bench_quoref_lite_studies_family(seed: int = _SEED + 3):
    """quoref_lite_studies: synthetic correctness bench."""
    return _finite_blob(quoref_lite_studies.bench_quoref_lite_studies(seed))


def bench_record_lite_studies_family(seed: int = _SEED + 4):
    """record_lite_studies: synthetic correctness bench."""
    return _finite_blob(record_lite_studies.bench_record_lite_studies(seed))


def bench_squad_lite2_studies_family(seed: int = _SEED + 5):
    """squad_lite2_studies: synthetic correctness bench."""
    return _finite_blob(squad_lite2_studies.bench_squad_lite2_studies(seed))
