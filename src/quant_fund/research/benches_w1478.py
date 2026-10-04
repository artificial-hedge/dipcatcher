"""Wave-1478 bench adapters: antelope canon (SYNTHETIC only)."""

from quant_fund.models import (
    antelope_qa_studies,
    eland_qa_studies,
    impala_qa_studies,
    kudu_qa_studies,
    oryx_qa_studies,
    springbok_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14780


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_antelope_qa_studies_family(seed: int = _SEED + 0):
    """antelope_qa_studies: synthetic correctness bench."""
    return _finite_blob(antelope_qa_studies.bench_antelope_qa_studies(seed))


def bench_eland_qa_studies_family(seed: int = _SEED + 1):
    """eland_qa_studies: synthetic correctness bench."""
    return _finite_blob(eland_qa_studies.bench_eland_qa_studies(seed))


def bench_impala_qa_studies_family(seed: int = _SEED + 2):
    """impala_qa_studies: synthetic correctness bench."""
    return _finite_blob(impala_qa_studies.bench_impala_qa_studies(seed))


def bench_kudu_qa_studies_family(seed: int = _SEED + 3):
    """kudu_qa_studies: synthetic correctness bench."""
    return _finite_blob(kudu_qa_studies.bench_kudu_qa_studies(seed))


def bench_oryx_qa_studies_family(seed: int = _SEED + 4):
    """oryx_qa_studies: synthetic correctness bench."""
    return _finite_blob(oryx_qa_studies.bench_oryx_qa_studies(seed))


def bench_springbok_qa_studies_family(seed: int = _SEED + 5):
    """springbok_qa_studies: synthetic correctness bench."""
    return _finite_blob(springbok_qa_studies.bench_springbok_qa_studies(seed))
