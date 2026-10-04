"""Wave-1329 bench adapters: long-context-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    gov_report_studies,
    looogle_studies,
    lost_middle_studies,
    marathon_eval_studies,
    niah_v2_studies,
    passkey_retrieval_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13290


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_gov_report_studies_family(seed: int = _SEED + 0):
    """gov_report_studies: synthetic correctness bench."""
    return _finite_blob(gov_report_studies.bench_gov_report_studies(seed))


def bench_looogle_studies_family(seed: int = _SEED + 1):
    """looogle_studies: synthetic correctness bench."""
    return _finite_blob(looogle_studies.bench_looogle_studies(seed))


def bench_lost_middle_studies_family(seed: int = _SEED + 2):
    """lost_middle_studies: synthetic correctness bench."""
    return _finite_blob(lost_middle_studies.bench_lost_middle_studies(seed))


def bench_marathon_eval_studies_family(seed: int = _SEED + 3):
    """marathon_eval_studies: synthetic correctness bench."""
    return _finite_blob(marathon_eval_studies.bench_marathon_eval_studies(seed))


def bench_niah_v2_studies_family(seed: int = _SEED + 4):
    """niah_v2_studies: synthetic correctness bench."""
    return _finite_blob(niah_v2_studies.bench_niah_v2_studies(seed))


def bench_passkey_retrieval_studies_family(seed: int = _SEED + 5):
    """passkey_retrieval_studies: synthetic correctness bench."""
    return _finite_blob(passkey_retrieval_studies.bench_passkey_retrieval_studies(seed))
