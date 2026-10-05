"""Wave-1941 bench adapters: khmer-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    ahp_qa_studies,
    arak_qa_studies,
    boramei_qa_studies,
    kmoch_qa_studies,
    mrenh_kongveal_qa_studies,
    neak_ta_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19410


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ahp_qa_studies_family(seed: int = _SEED + 0):
    """ahp_qa_studies: synthetic correctness bench."""
    return _finite_blob(ahp_qa_studies.bench_ahp_qa_studies(seed))


def bench_arak_qa_studies_family(seed: int = _SEED + 1):
    """arak_qa_studies: synthetic correctness bench."""
    return _finite_blob(arak_qa_studies.bench_arak_qa_studies(seed))


def bench_boramei_qa_studies_family(seed: int = _SEED + 2):
    """boramei_qa_studies: synthetic correctness bench."""
    return _finite_blob(boramei_qa_studies.bench_boramei_qa_studies(seed))


def bench_kmoch_qa_studies_family(seed: int = _SEED + 3):
    """kmoch_qa_studies: synthetic correctness bench."""
    return _finite_blob(kmoch_qa_studies.bench_kmoch_qa_studies(seed))


def bench_mrenh_kongveal_qa_studies_family(seed: int = _SEED + 4):
    """mrenh_kongveal_qa_studies: synthetic correctness bench."""
    return _finite_blob(mrenh_kongveal_qa_studies.bench_mrenh_kongveal_qa_studies(seed))


def bench_neak_ta_qa_studies_family(seed: int = _SEED + 5):
    """neak_ta_qa_studies: synthetic correctness bench."""
    return _finite_blob(neak_ta_qa_studies.bench_neak_ta_qa_studies(seed))
