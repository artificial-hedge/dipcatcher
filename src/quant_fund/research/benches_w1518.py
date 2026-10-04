"""Wave-1518 bench adapters: hummingbird-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    coquette_qa_studies,
    fairy_qa_studies,
    jacobin_qa_studies,
    lancebill_qa_studies,
    sabrewing_qa_studies,
    sheartail_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15180


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_coquette_qa_studies_family(seed: int = _SEED + 0):
    """coquette_qa_studies: synthetic correctness bench."""
    return _finite_blob(coquette_qa_studies.bench_coquette_qa_studies(seed))


def bench_fairy_qa_studies_family(seed: int = _SEED + 1):
    """fairy_qa_studies: synthetic correctness bench."""
    return _finite_blob(fairy_qa_studies.bench_fairy_qa_studies(seed))


def bench_jacobin_qa_studies_family(seed: int = _SEED + 2):
    """jacobin_qa_studies: synthetic correctness bench."""
    return _finite_blob(jacobin_qa_studies.bench_jacobin_qa_studies(seed))


def bench_lancebill_qa_studies_family(seed: int = _SEED + 3):
    """lancebill_qa_studies: synthetic correctness bench."""
    return _finite_blob(lancebill_qa_studies.bench_lancebill_qa_studies(seed))


def bench_sabrewing_qa_studies_family(seed: int = _SEED + 4):
    """sabrewing_qa_studies: synthetic correctness bench."""
    return _finite_blob(sabrewing_qa_studies.bench_sabrewing_qa_studies(seed))


def bench_sheartail_qa_studies_family(seed: int = _SEED + 5):
    """sheartail_qa_studies: synthetic correctness bench."""
    return _finite_blob(sheartail_qa_studies.bench_sheartail_qa_studies(seed))
