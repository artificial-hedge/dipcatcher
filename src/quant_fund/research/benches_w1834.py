"""Wave-1834 bench adapters: hurrian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    hebat2_qa_studies,
    kusuh2_qa_studies,
    sarruma2_qa_studies,
    simige2_qa_studies,
    tasmisu2_qa_studies,
    tessub2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18340


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_hebat2_qa_studies_family(seed: int = _SEED + 0):
    """hebat2_qa_studies: synthetic correctness bench."""
    return _finite_blob(hebat2_qa_studies.bench_hebat2_qa_studies(seed))


def bench_kusuh2_qa_studies_family(seed: int = _SEED + 1):
    """kusuh2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kusuh2_qa_studies.bench_kusuh2_qa_studies(seed))


def bench_sarruma2_qa_studies_family(seed: int = _SEED + 2):
    """sarruma2_qa_studies: synthetic correctness bench."""
    return _finite_blob(sarruma2_qa_studies.bench_sarruma2_qa_studies(seed))


def bench_simige2_qa_studies_family(seed: int = _SEED + 3):
    """simige2_qa_studies: synthetic correctness bench."""
    return _finite_blob(simige2_qa_studies.bench_simige2_qa_studies(seed))


def bench_tasmisu2_qa_studies_family(seed: int = _SEED + 4):
    """tasmisu2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tasmisu2_qa_studies.bench_tasmisu2_qa_studies(seed))


def bench_tessub2_qa_studies_family(seed: int = _SEED + 5):
    """tessub2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tessub2_qa_studies.bench_tessub2_qa_studies(seed))
