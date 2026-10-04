"""Wave-1847 bench adapters: himyarite-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    dhatanwat_qa_studies,
    dhatzahran_qa_studies,
    hawl_qa_studies,
    khalasah_qa_studies,
    raymah_qa_studies,
    shams_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18470


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_dhatanwat_qa_studies_family(seed: int = _SEED + 0):
    """dhatanwat_qa_studies: synthetic correctness bench."""
    return _finite_blob(dhatanwat_qa_studies.bench_dhatanwat_qa_studies(seed))


def bench_dhatzahran_qa_studies_family(seed: int = _SEED + 1):
    """dhatzahran_qa_studies: synthetic correctness bench."""
    return _finite_blob(dhatzahran_qa_studies.bench_dhatzahran_qa_studies(seed))


def bench_hawl_qa_studies_family(seed: int = _SEED + 2):
    """hawl_qa_studies: synthetic correctness bench."""
    return _finite_blob(hawl_qa_studies.bench_hawl_qa_studies(seed))


def bench_khalasah_qa_studies_family(seed: int = _SEED + 3):
    """khalasah_qa_studies: synthetic correctness bench."""
    return _finite_blob(khalasah_qa_studies.bench_khalasah_qa_studies(seed))


def bench_raymah_qa_studies_family(seed: int = _SEED + 4):
    """raymah_qa_studies: synthetic correctness bench."""
    return _finite_blob(raymah_qa_studies.bench_raymah_qa_studies(seed))


def bench_shams_qa_studies_family(seed: int = _SEED + 5):
    """shams_qa_studies: synthetic correctness bench."""
    return _finite_blob(shams_qa_studies.bench_shams_qa_studies(seed))
