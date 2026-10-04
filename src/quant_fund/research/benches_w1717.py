"""Wave-1717 bench adapters: illyrian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    bindus_qa_studies,
    illyris_qa_studies,
    medaurus_qa_studies,
    redon_qa_studies,
    thana_qa_studies,
    vidasus_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17170


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bindus_qa_studies_family(seed: int = _SEED + 0):
    """bindus_qa_studies: synthetic correctness bench."""
    return _finite_blob(bindus_qa_studies.bench_bindus_qa_studies(seed))


def bench_illyris_qa_studies_family(seed: int = _SEED + 1):
    """illyris_qa_studies: synthetic correctness bench."""
    return _finite_blob(illyris_qa_studies.bench_illyris_qa_studies(seed))


def bench_medaurus_qa_studies_family(seed: int = _SEED + 2):
    """medaurus_qa_studies: synthetic correctness bench."""
    return _finite_blob(medaurus_qa_studies.bench_medaurus_qa_studies(seed))


def bench_redon_qa_studies_family(seed: int = _SEED + 3):
    """redon_qa_studies: synthetic correctness bench."""
    return _finite_blob(redon_qa_studies.bench_redon_qa_studies(seed))


def bench_thana_qa_studies_family(seed: int = _SEED + 4):
    """thana_qa_studies: synthetic correctness bench."""
    return _finite_blob(thana_qa_studies.bench_thana_qa_studies(seed))


def bench_vidasus_qa_studies_family(seed: int = _SEED + 5):
    """vidasus_qa_studies: synthetic correctness bench."""
    return _finite_blob(vidasus_qa_studies.bench_vidasus_qa_studies(seed))
