"""Wave-1866 bench adapters: arthurian-7 canon (SYNTHETIC only)."""

from quant_fund.models import (
    avilion_qa_studies,
    balan_qa_studies,
    ector_qa_studies,
    hector_cameliard_qa_studies,
    seneschal_qa_studies,
    ynis_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18660


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_avilion_qa_studies_family(seed: int = _SEED + 0):
    """avilion_qa_studies: synthetic correctness bench."""
    return _finite_blob(avilion_qa_studies.bench_avilion_qa_studies(seed))


def bench_balan_qa_studies_family(seed: int = _SEED + 1):
    """balan_qa_studies: synthetic correctness bench."""
    return _finite_blob(balan_qa_studies.bench_balan_qa_studies(seed))


def bench_ector_qa_studies_family(seed: int = _SEED + 2):
    """ector_qa_studies: synthetic correctness bench."""
    return _finite_blob(ector_qa_studies.bench_ector_qa_studies(seed))


def bench_hector_cameliard_qa_studies_family(seed: int = _SEED + 3):
    """hector_cameliard_qa_studies: synthetic correctness bench."""
    return _finite_blob(hector_cameliard_qa_studies.bench_hector_cameliard_qa_studies(seed))


def bench_seneschal_qa_studies_family(seed: int = _SEED + 4):
    """seneschal_qa_studies: synthetic correctness bench."""
    return _finite_blob(seneschal_qa_studies.bench_seneschal_qa_studies(seed))


def bench_ynis_qa_studies_family(seed: int = _SEED + 5):
    """ynis_qa_studies: synthetic correctness bench."""
    return _finite_blob(ynis_qa_studies.bench_ynis_qa_studies(seed))
