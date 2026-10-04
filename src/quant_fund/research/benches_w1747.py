"""Wave-1747 bench adapters: norse-myth-6 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bragi_qa_studies,
    forseti_qa_studies,
    idun_qa_studies,
    sif_qa_studies,
    ullr_qa_studies,
    vidar_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17470


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bragi_qa_studies_family(seed: int = _SEED + 0):
    """bragi_qa_studies: synthetic correctness bench."""
    return _finite_blob(bragi_qa_studies.bench_bragi_qa_studies(seed))


def bench_forseti_qa_studies_family(seed: int = _SEED + 1):
    """forseti_qa_studies: synthetic correctness bench."""
    return _finite_blob(forseti_qa_studies.bench_forseti_qa_studies(seed))


def bench_idun_qa_studies_family(seed: int = _SEED + 2):
    """idun_qa_studies: synthetic correctness bench."""
    return _finite_blob(idun_qa_studies.bench_idun_qa_studies(seed))


def bench_sif_qa_studies_family(seed: int = _SEED + 3):
    """sif_qa_studies: synthetic correctness bench."""
    return _finite_blob(sif_qa_studies.bench_sif_qa_studies(seed))


def bench_ullr_qa_studies_family(seed: int = _SEED + 4):
    """ullr_qa_studies: synthetic correctness bench."""
    return _finite_blob(ullr_qa_studies.bench_ullr_qa_studies(seed))


def bench_vidar_qa_studies_family(seed: int = _SEED + 5):
    """vidar_qa_studies: synthetic correctness bench."""
    return _finite_blob(vidar_qa_studies.bench_vidar_qa_studies(seed))
