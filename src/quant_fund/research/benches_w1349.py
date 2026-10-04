"""Wave-1349 bench adapters: GLUE-eval-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    cola_lite_studies,
    qnli_lite_studies,
    qqp_lite_studies,
    sst2_lite_studies,
    stsb_lite_studies,
    wnli_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13490


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cola_lite_studies_family(seed: int = _SEED + 0):
    """cola_lite_studies: synthetic correctness bench."""
    return _finite_blob(cola_lite_studies.bench_cola_lite_studies(seed))


def bench_qnli_lite_studies_family(seed: int = _SEED + 1):
    """qnli_lite_studies: synthetic correctness bench."""
    return _finite_blob(qnli_lite_studies.bench_qnli_lite_studies(seed))


def bench_qqp_lite_studies_family(seed: int = _SEED + 2):
    """qqp_lite_studies: synthetic correctness bench."""
    return _finite_blob(qqp_lite_studies.bench_qqp_lite_studies(seed))


def bench_sst2_lite_studies_family(seed: int = _SEED + 3):
    """sst2_lite_studies: synthetic correctness bench."""
    return _finite_blob(sst2_lite_studies.bench_sst2_lite_studies(seed))


def bench_stsb_lite_studies_family(seed: int = _SEED + 4):
    """stsb_lite_studies: synthetic correctness bench."""
    return _finite_blob(stsb_lite_studies.bench_stsb_lite_studies(seed))


def bench_wnli_lite_studies_family(seed: int = _SEED + 5):
    """wnli_lite_studies: synthetic correctness bench."""
    return _finite_blob(wnli_lite_studies.bench_wnli_lite_studies(seed))
