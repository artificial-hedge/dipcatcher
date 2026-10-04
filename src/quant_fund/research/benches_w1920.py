"""Wave-1920 bench adapters: romanian-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    iele_qa_studies,
    moroi_qa_studies,
    pricolici_qa_studies,
    samca_qa_studies,
    strigoi_qa_studies,
    varcolac_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19200


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_iele_qa_studies_family(seed: int = _SEED + 0):
    """iele_qa_studies: synthetic correctness bench."""
    return _finite_blob(iele_qa_studies.bench_iele_qa_studies(seed))


def bench_moroi_qa_studies_family(seed: int = _SEED + 1):
    """moroi_qa_studies: synthetic correctness bench."""
    return _finite_blob(moroi_qa_studies.bench_moroi_qa_studies(seed))


def bench_pricolici_qa_studies_family(seed: int = _SEED + 2):
    """pricolici_qa_studies: synthetic correctness bench."""
    return _finite_blob(pricolici_qa_studies.bench_pricolici_qa_studies(seed))


def bench_samca_qa_studies_family(seed: int = _SEED + 3):
    """samca_qa_studies: synthetic correctness bench."""
    return _finite_blob(samca_qa_studies.bench_samca_qa_studies(seed))


def bench_strigoi_qa_studies_family(seed: int = _SEED + 4):
    """strigoi_qa_studies: synthetic correctness bench."""
    return _finite_blob(strigoi_qa_studies.bench_strigoi_qa_studies(seed))


def bench_varcolac_qa_studies_family(seed: int = _SEED + 5):
    """varcolac_qa_studies: synthetic correctness bench."""
    return _finite_blob(varcolac_qa_studies.bench_varcolac_qa_studies(seed))
