"""Wave-1956 bench adapters: goetic-covenant canon (SYNTHETIC only)."""

from quant_fund.models import (
    focalor_qa_studies,
    halphas_qa_studies,
    raum_qa_studies,
    sabnock_qa_studies,
    shax_qa_studies,
    vepar_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19560


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_focalor_qa_studies_family(seed: int = _SEED + 0):
    """focalor_qa_studies: synthetic correctness bench."""
    return _finite_blob(focalor_qa_studies.bench_focalor_qa_studies(seed))


def bench_halphas_qa_studies_family(seed: int = _SEED + 1):
    """halphas_qa_studies: synthetic correctness bench."""
    return _finite_blob(halphas_qa_studies.bench_halphas_qa_studies(seed))


def bench_raum_qa_studies_family(seed: int = _SEED + 2):
    """raum_qa_studies: synthetic correctness bench."""
    return _finite_blob(raum_qa_studies.bench_raum_qa_studies(seed))


def bench_sabnock_qa_studies_family(seed: int = _SEED + 3):
    """sabnock_qa_studies: synthetic correctness bench."""
    return _finite_blob(sabnock_qa_studies.bench_sabnock_qa_studies(seed))


def bench_shax_qa_studies_family(seed: int = _SEED + 4):
    """shax_qa_studies: synthetic correctness bench."""
    return _finite_blob(shax_qa_studies.bench_shax_qa_studies(seed))


def bench_vepar_qa_studies_family(seed: int = _SEED + 5):
    """vepar_qa_studies: synthetic correctness bench."""
    return _finite_blob(vepar_qa_studies.bench_vepar_qa_studies(seed))
