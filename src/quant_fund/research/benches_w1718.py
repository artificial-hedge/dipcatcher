"""Wave-1718 bench adapters: etruscan-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    fufluns_qa_studies,
    menrva_qa_studies,
    tinia_qa_studies,
    turan_qa_studies,
    veltha_qa_studies,
    voltumna_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17180


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_fufluns_qa_studies_family(seed: int = _SEED + 0):
    """fufluns_qa_studies: synthetic correctness bench."""
    return _finite_blob(fufluns_qa_studies.bench_fufluns_qa_studies(seed))


def bench_menrva_qa_studies_family(seed: int = _SEED + 1):
    """menrva_qa_studies: synthetic correctness bench."""
    return _finite_blob(menrva_qa_studies.bench_menrva_qa_studies(seed))


def bench_tinia_qa_studies_family(seed: int = _SEED + 2):
    """tinia_qa_studies: synthetic correctness bench."""
    return _finite_blob(tinia_qa_studies.bench_tinia_qa_studies(seed))


def bench_turan_qa_studies_family(seed: int = _SEED + 3):
    """turan_qa_studies: synthetic correctness bench."""
    return _finite_blob(turan_qa_studies.bench_turan_qa_studies(seed))


def bench_veltha_qa_studies_family(seed: int = _SEED + 4):
    """veltha_qa_studies: synthetic correctness bench."""
    return _finite_blob(veltha_qa_studies.bench_veltha_qa_studies(seed))


def bench_voltumna_qa_studies_family(seed: int = _SEED + 5):
    """voltumna_qa_studies: synthetic correctness bench."""
    return _finite_blob(voltumna_qa_studies.bench_voltumna_qa_studies(seed))
