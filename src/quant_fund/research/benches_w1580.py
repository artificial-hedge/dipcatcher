"""Wave-1580 bench adapters: deer canon (SYNTHETIC only)."""

from quant_fund.models import (
    chital_qa_studies,
    fallow_qa_studies,
    muntjac_qa_studies,
    pudu_qa_studies,
    roe_qa_studies,
    sika_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15800


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_chital_qa_studies_family(seed: int = _SEED + 0):
    """chital_qa_studies: synthetic correctness bench."""
    return _finite_blob(chital_qa_studies.bench_chital_qa_studies(seed))


def bench_fallow_qa_studies_family(seed: int = _SEED + 1):
    """fallow_qa_studies: synthetic correctness bench."""
    return _finite_blob(fallow_qa_studies.bench_fallow_qa_studies(seed))


def bench_muntjac_qa_studies_family(seed: int = _SEED + 2):
    """muntjac_qa_studies: synthetic correctness bench."""
    return _finite_blob(muntjac_qa_studies.bench_muntjac_qa_studies(seed))


def bench_pudu_qa_studies_family(seed: int = _SEED + 3):
    """pudu_qa_studies: synthetic correctness bench."""
    return _finite_blob(pudu_qa_studies.bench_pudu_qa_studies(seed))


def bench_roe_qa_studies_family(seed: int = _SEED + 4):
    """roe_qa_studies: synthetic correctness bench."""
    return _finite_blob(roe_qa_studies.bench_roe_qa_studies(seed))


def bench_sika_qa_studies_family(seed: int = _SEED + 5):
    """sika_qa_studies: synthetic correctness bench."""
    return _finite_blob(sika_qa_studies.bench_sika_qa_studies(seed))
