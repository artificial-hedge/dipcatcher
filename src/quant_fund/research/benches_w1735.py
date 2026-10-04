"""Wave-1735 bench adapters: lithuanian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    dievas_qa_studies,
    gabija_qa_studies,
    medeina_qa_studies,
    ragana_qa_studies,
    saulute_qa_studies,
    velnias_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17350


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_dievas_qa_studies_family(seed: int = _SEED + 0):
    """dievas_qa_studies: synthetic correctness bench."""
    return _finite_blob(dievas_qa_studies.bench_dievas_qa_studies(seed))


def bench_gabija_qa_studies_family(seed: int = _SEED + 1):
    """gabija_qa_studies: synthetic correctness bench."""
    return _finite_blob(gabija_qa_studies.bench_gabija_qa_studies(seed))


def bench_medeina_qa_studies_family(seed: int = _SEED + 2):
    """medeina_qa_studies: synthetic correctness bench."""
    return _finite_blob(medeina_qa_studies.bench_medeina_qa_studies(seed))


def bench_ragana_qa_studies_family(seed: int = _SEED + 3):
    """ragana_qa_studies: synthetic correctness bench."""
    return _finite_blob(ragana_qa_studies.bench_ragana_qa_studies(seed))


def bench_saulute_qa_studies_family(seed: int = _SEED + 4):
    """saulute_qa_studies: synthetic correctness bench."""
    return _finite_blob(saulute_qa_studies.bench_saulute_qa_studies(seed))


def bench_velnias_qa_studies_family(seed: int = _SEED + 5):
    """velnias_qa_studies: synthetic correctness bench."""
    return _finite_blob(velnias_qa_studies.bench_velnias_qa_studies(seed))
