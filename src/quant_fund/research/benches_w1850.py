"""Wave-1850 bench adapters: amazigh-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    amma_qa_studies,
    anzar_qa_studies,
    ayyur_qa_studies,
    ifri_qa_studies,
    meghisen_qa_studies,
    tanit2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18500


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amma_qa_studies_family(seed: int = _SEED + 0):
    """amma_qa_studies: synthetic correctness bench."""
    return _finite_blob(amma_qa_studies.bench_amma_qa_studies(seed))


def bench_anzar_qa_studies_family(seed: int = _SEED + 1):
    """anzar_qa_studies: synthetic correctness bench."""
    return _finite_blob(anzar_qa_studies.bench_anzar_qa_studies(seed))


def bench_ayyur_qa_studies_family(seed: int = _SEED + 2):
    """ayyur_qa_studies: synthetic correctness bench."""
    return _finite_blob(ayyur_qa_studies.bench_ayyur_qa_studies(seed))


def bench_ifri_qa_studies_family(seed: int = _SEED + 3):
    """ifri_qa_studies: synthetic correctness bench."""
    return _finite_blob(ifri_qa_studies.bench_ifri_qa_studies(seed))


def bench_meghisen_qa_studies_family(seed: int = _SEED + 4):
    """meghisen_qa_studies: synthetic correctness bench."""
    return _finite_blob(meghisen_qa_studies.bench_meghisen_qa_studies(seed))


def bench_tanit2_qa_studies_family(seed: int = _SEED + 5):
    """tanit2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tanit2_qa_studies.bench_tanit2_qa_studies(seed))
