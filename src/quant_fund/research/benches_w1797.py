"""Wave-1797 bench adapters: greek-myth-10 canon (SYNTHETIC only)."""

from quant_fund.models import (
    atropos_qa_studies,
    dikaion_qa_studies,
    eunomia_qa_studies,
    lachesis_qa_studies,
    metis2_qa_studies,
    peitho_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17970


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_atropos_qa_studies_family(seed: int = _SEED + 0):
    """atropos_qa_studies: synthetic correctness bench."""
    return _finite_blob(atropos_qa_studies.bench_atropos_qa_studies(seed))


def bench_dikaion_qa_studies_family(seed: int = _SEED + 1):
    """dikaion_qa_studies: synthetic correctness bench."""
    return _finite_blob(dikaion_qa_studies.bench_dikaion_qa_studies(seed))


def bench_eunomia_qa_studies_family(seed: int = _SEED + 2):
    """eunomia_qa_studies: synthetic correctness bench."""
    return _finite_blob(eunomia_qa_studies.bench_eunomia_qa_studies(seed))


def bench_lachesis_qa_studies_family(seed: int = _SEED + 3):
    """lachesis_qa_studies: synthetic correctness bench."""
    return _finite_blob(lachesis_qa_studies.bench_lachesis_qa_studies(seed))


def bench_metis2_qa_studies_family(seed: int = _SEED + 4):
    """metis2_qa_studies: synthetic correctness bench."""
    return _finite_blob(metis2_qa_studies.bench_metis2_qa_studies(seed))


def bench_peitho_qa_studies_family(seed: int = _SEED + 5):
    """peitho_qa_studies: synthetic correctness bench."""
    return _finite_blob(peitho_qa_studies.bench_peitho_qa_studies(seed))
