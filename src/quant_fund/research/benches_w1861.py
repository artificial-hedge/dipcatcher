"""Wave-1861 bench adapters: arthurian-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bedivere_qa_studies,
    galahad_qa_studies,
    gawain_qa_studies,
    lancelot_qa_studies,
    percival_qa_studies,
    tristan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18610


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bedivere_qa_studies_family(seed: int = _SEED + 0):
    """bedivere_qa_studies: synthetic correctness bench."""
    return _finite_blob(bedivere_qa_studies.bench_bedivere_qa_studies(seed))


def bench_galahad_qa_studies_family(seed: int = _SEED + 1):
    """galahad_qa_studies: synthetic correctness bench."""
    return _finite_blob(galahad_qa_studies.bench_galahad_qa_studies(seed))


def bench_gawain_qa_studies_family(seed: int = _SEED + 2):
    """gawain_qa_studies: synthetic correctness bench."""
    return _finite_blob(gawain_qa_studies.bench_gawain_qa_studies(seed))


def bench_lancelot_qa_studies_family(seed: int = _SEED + 3):
    """lancelot_qa_studies: synthetic correctness bench."""
    return _finite_blob(lancelot_qa_studies.bench_lancelot_qa_studies(seed))


def bench_percival_qa_studies_family(seed: int = _SEED + 4):
    """percival_qa_studies: synthetic correctness bench."""
    return _finite_blob(percival_qa_studies.bench_percival_qa_studies(seed))


def bench_tristan_qa_studies_family(seed: int = _SEED + 5):
    """tristan_qa_studies: synthetic correctness bench."""
    return _finite_blob(tristan_qa_studies.bench_tristan_qa_studies(seed))
