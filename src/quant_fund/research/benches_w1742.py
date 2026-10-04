"""Wave-1742 bench adapters: maori-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    haumia_qa_studies,
    rongo_qa_studies,
    tane_qa_studies,
    tangaroa_qa_studies,
    tawhirimatea_qa_studies,
    tumatauenga_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17420


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_haumia_qa_studies_family(seed: int = _SEED + 0):
    """haumia_qa_studies: synthetic correctness bench."""
    return _finite_blob(haumia_qa_studies.bench_haumia_qa_studies(seed))


def bench_rongo_qa_studies_family(seed: int = _SEED + 1):
    """rongo_qa_studies: synthetic correctness bench."""
    return _finite_blob(rongo_qa_studies.bench_rongo_qa_studies(seed))


def bench_tane_qa_studies_family(seed: int = _SEED + 2):
    """tane_qa_studies: synthetic correctness bench."""
    return _finite_blob(tane_qa_studies.bench_tane_qa_studies(seed))


def bench_tangaroa_qa_studies_family(seed: int = _SEED + 3):
    """tangaroa_qa_studies: synthetic correctness bench."""
    return _finite_blob(tangaroa_qa_studies.bench_tangaroa_qa_studies(seed))


def bench_tawhirimatea_qa_studies_family(seed: int = _SEED + 4):
    """tawhirimatea_qa_studies: synthetic correctness bench."""
    return _finite_blob(tawhirimatea_qa_studies.bench_tawhirimatea_qa_studies(seed))


def bench_tumatauenga_qa_studies_family(seed: int = _SEED + 5):
    """tumatauenga_qa_studies: synthetic correctness bench."""
    return _finite_blob(tumatauenga_qa_studies.bench_tumatauenga_qa_studies(seed))
