"""Wave-1652 bench adapters: european-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    cuco_qa_studies,
    dahu_qa_studies,
    gargouille_qa_studies,
    lavellan_qa_studies,
    muscaliet_qa_studies,
    tarasque_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16520


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cuco_qa_studies_family(seed: int = _SEED + 0):
    """cuco_qa_studies: synthetic correctness bench."""
    return _finite_blob(cuco_qa_studies.bench_cuco_qa_studies(seed))


def bench_dahu_qa_studies_family(seed: int = _SEED + 1):
    """dahu_qa_studies: synthetic correctness bench."""
    return _finite_blob(dahu_qa_studies.bench_dahu_qa_studies(seed))


def bench_gargouille_qa_studies_family(seed: int = _SEED + 2):
    """gargouille_qa_studies: synthetic correctness bench."""
    return _finite_blob(gargouille_qa_studies.bench_gargouille_qa_studies(seed))


def bench_lavellan_qa_studies_family(seed: int = _SEED + 3):
    """lavellan_qa_studies: synthetic correctness bench."""
    return _finite_blob(lavellan_qa_studies.bench_lavellan_qa_studies(seed))


def bench_muscaliet_qa_studies_family(seed: int = _SEED + 4):
    """muscaliet_qa_studies: synthetic correctness bench."""
    return _finite_blob(muscaliet_qa_studies.bench_muscaliet_qa_studies(seed))


def bench_tarasque_qa_studies_family(seed: int = _SEED + 5):
    """tarasque_qa_studies: synthetic correctness bench."""
    return _finite_blob(tarasque_qa_studies.bench_tarasque_qa_studies(seed))
