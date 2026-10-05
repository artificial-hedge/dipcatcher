"""Wave-1957 bench adapters: goetic-sigil canon (SYNTHETIC only)."""

from quant_fund.models import (
    bael_qa_studies,
    bifrons_qa_studies,
    crocell_qa_studies,
    gamigin_qa_studies,
    haagenti_qa_studies,
    vuall_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19570


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bael_qa_studies_family(seed: int = _SEED + 0):
    """bael_qa_studies: synthetic correctness bench."""
    return _finite_blob(bael_qa_studies.bench_bael_qa_studies(seed))


def bench_bifrons_qa_studies_family(seed: int = _SEED + 1):
    """bifrons_qa_studies: synthetic correctness bench."""
    return _finite_blob(bifrons_qa_studies.bench_bifrons_qa_studies(seed))


def bench_crocell_qa_studies_family(seed: int = _SEED + 2):
    """crocell_qa_studies: synthetic correctness bench."""
    return _finite_blob(crocell_qa_studies.bench_crocell_qa_studies(seed))


def bench_gamigin_qa_studies_family(seed: int = _SEED + 3):
    """gamigin_qa_studies: synthetic correctness bench."""
    return _finite_blob(gamigin_qa_studies.bench_gamigin_qa_studies(seed))


def bench_haagenti_qa_studies_family(seed: int = _SEED + 4):
    """haagenti_qa_studies: synthetic correctness bench."""
    return _finite_blob(haagenti_qa_studies.bench_haagenti_qa_studies(seed))


def bench_vuall_qa_studies_family(seed: int = _SEED + 5):
    """vuall_qa_studies: synthetic correctness bench."""
    return _finite_blob(vuall_qa_studies.bench_vuall_qa_studies(seed))
