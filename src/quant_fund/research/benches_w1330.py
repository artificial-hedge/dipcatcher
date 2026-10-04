"""Wave-1330 bench adapters: safety-alignment-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    beaver_safe_studies,
    do_not_answer_studies,
    hh_rlhf_studies,
    honest_eval_studies,
    safe_rlhf_studies,
    sos_bench_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13300


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_beaver_safe_studies_family(seed: int = _SEED + 0):
    """beaver_safe_studies: synthetic correctness bench."""
    return _finite_blob(beaver_safe_studies.bench_beaver_safe_studies(seed))


def bench_do_not_answer_studies_family(seed: int = _SEED + 1):
    """do_not_answer_studies: synthetic correctness bench."""
    return _finite_blob(do_not_answer_studies.bench_do_not_answer_studies(seed))


def bench_hh_rlhf_studies_family(seed: int = _SEED + 2):
    """hh_rlhf_studies: synthetic correctness bench."""
    return _finite_blob(hh_rlhf_studies.bench_hh_rlhf_studies(seed))


def bench_honest_eval_studies_family(seed: int = _SEED + 3):
    """honest_eval_studies: synthetic correctness bench."""
    return _finite_blob(honest_eval_studies.bench_honest_eval_studies(seed))


def bench_safe_rlhf_studies_family(seed: int = _SEED + 4):
    """safe_rlhf_studies: synthetic correctness bench."""
    return _finite_blob(safe_rlhf_studies.bench_safe_rlhf_studies(seed))


def bench_sos_bench_studies_family(seed: int = _SEED + 5):
    """sos_bench_studies: synthetic correctness bench."""
    return _finite_blob(sos_bench_studies.bench_sos_bench_studies(seed))
