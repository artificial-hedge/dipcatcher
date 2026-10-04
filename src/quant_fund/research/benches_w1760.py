"""Wave-1760 bench adapters: japanese-myth-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    ebisu_qa_studies,
    hiruko_qa_studies,
    kikuzuki_qa_studies,
    kisshoten_qa_studies,
    morinaga_qa_studies,
    senju_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17600


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ebisu_qa_studies_family(seed: int = _SEED + 0):
    """ebisu_qa_studies: synthetic correctness bench."""
    return _finite_blob(ebisu_qa_studies.bench_ebisu_qa_studies(seed))


def bench_hiruko_qa_studies_family(seed: int = _SEED + 1):
    """hiruko_qa_studies: synthetic correctness bench."""
    return _finite_blob(hiruko_qa_studies.bench_hiruko_qa_studies(seed))


def bench_kikuzuki_qa_studies_family(seed: int = _SEED + 2):
    """kikuzuki_qa_studies: synthetic correctness bench."""
    return _finite_blob(kikuzuki_qa_studies.bench_kikuzuki_qa_studies(seed))


def bench_kisshoten_qa_studies_family(seed: int = _SEED + 3):
    """kisshoten_qa_studies: synthetic correctness bench."""
    return _finite_blob(kisshoten_qa_studies.bench_kisshoten_qa_studies(seed))


def bench_morinaga_qa_studies_family(seed: int = _SEED + 4):
    """morinaga_qa_studies: synthetic correctness bench."""
    return _finite_blob(morinaga_qa_studies.bench_morinaga_qa_studies(seed))


def bench_senju_qa_studies_family(seed: int = _SEED + 5):
    """senju_qa_studies: synthetic correctness bench."""
    return _finite_blob(senju_qa_studies.bench_senju_qa_studies(seed))
