"""Wave-1874 bench adapters: manx-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    arkan_sonney_qa_studies,
    dozmary_qa_studies,
    loaghtan_qa_studies,
    shooil_ghoul_qa_studies,
    sleih_beggey_qa_studies,
    ushtey_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18740


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_arkan_sonney_qa_studies_family(seed: int = _SEED + 0):
    """arkan_sonney_qa_studies: synthetic correctness bench."""
    return _finite_blob(arkan_sonney_qa_studies.bench_arkan_sonney_qa_studies(seed))


def bench_dozmary_qa_studies_family(seed: int = _SEED + 1):
    """dozmary_qa_studies: synthetic correctness bench."""
    return _finite_blob(dozmary_qa_studies.bench_dozmary_qa_studies(seed))


def bench_loaghtan_qa_studies_family(seed: int = _SEED + 2):
    """loaghtan_qa_studies: synthetic correctness bench."""
    return _finite_blob(loaghtan_qa_studies.bench_loaghtan_qa_studies(seed))


def bench_shooil_ghoul_qa_studies_family(seed: int = _SEED + 3):
    """shooil_ghoul_qa_studies: synthetic correctness bench."""
    return _finite_blob(shooil_ghoul_qa_studies.bench_shooil_ghoul_qa_studies(seed))


def bench_sleih_beggey_qa_studies_family(seed: int = _SEED + 4):
    """sleih_beggey_qa_studies: synthetic correctness bench."""
    return _finite_blob(sleih_beggey_qa_studies.bench_sleih_beggey_qa_studies(seed))


def bench_ushtey_qa_studies_family(seed: int = _SEED + 5):
    """ushtey_qa_studies: synthetic correctness bench."""
    return _finite_blob(ushtey_qa_studies.bench_ushtey_qa_studies(seed))
