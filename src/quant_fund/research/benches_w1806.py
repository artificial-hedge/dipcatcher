"""Wave-1806 bench adapters: persian-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    ahura2_qa_studies,
    ameretat2_qa_studies,
    atar2_qa_studies,
    haurvatat2_qa_studies,
    spenta2_qa_studies,
    verethragna2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18060


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ahura2_qa_studies_family(seed: int = _SEED + 0):
    """ahura2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ahura2_qa_studies.bench_ahura2_qa_studies(seed))


def bench_ameretat2_qa_studies_family(seed: int = _SEED + 1):
    """ameretat2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ameretat2_qa_studies.bench_ameretat2_qa_studies(seed))


def bench_atar2_qa_studies_family(seed: int = _SEED + 2):
    """atar2_qa_studies: synthetic correctness bench."""
    return _finite_blob(atar2_qa_studies.bench_atar2_qa_studies(seed))


def bench_haurvatat2_qa_studies_family(seed: int = _SEED + 3):
    """haurvatat2_qa_studies: synthetic correctness bench."""
    return _finite_blob(haurvatat2_qa_studies.bench_haurvatat2_qa_studies(seed))


def bench_spenta2_qa_studies_family(seed: int = _SEED + 4):
    """spenta2_qa_studies: synthetic correctness bench."""
    return _finite_blob(spenta2_qa_studies.bench_spenta2_qa_studies(seed))


def bench_verethragna2_qa_studies_family(seed: int = _SEED + 5):
    """verethragna2_qa_studies: synthetic correctness bench."""
    return _finite_blob(verethragna2_qa_studies.bench_verethragna2_qa_studies(seed))
