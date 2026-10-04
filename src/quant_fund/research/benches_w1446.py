"""Wave-1446 bench adapters: instrument canon (SYNTHETIC only)."""

from quant_fund.models import (
    cello_qa_studies,
    drum_qa_studies,
    flute_qa_studies,
    guitar_qa_studies,
    piano_qa_studies,
    violin_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14460


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cello_qa_studies_family(seed: int = _SEED + 0):
    """cello_qa_studies: synthetic correctness bench."""
    return _finite_blob(cello_qa_studies.bench_cello_qa_studies(seed))


def bench_drum_qa_studies_family(seed: int = _SEED + 1):
    """drum_qa_studies: synthetic correctness bench."""
    return _finite_blob(drum_qa_studies.bench_drum_qa_studies(seed))


def bench_flute_qa_studies_family(seed: int = _SEED + 2):
    """flute_qa_studies: synthetic correctness bench."""
    return _finite_blob(flute_qa_studies.bench_flute_qa_studies(seed))


def bench_guitar_qa_studies_family(seed: int = _SEED + 3):
    """guitar_qa_studies: synthetic correctness bench."""
    return _finite_blob(guitar_qa_studies.bench_guitar_qa_studies(seed))


def bench_piano_qa_studies_family(seed: int = _SEED + 4):
    """piano_qa_studies: synthetic correctness bench."""
    return _finite_blob(piano_qa_studies.bench_piano_qa_studies(seed))


def bench_violin_qa_studies_family(seed: int = _SEED + 5):
    """violin_qa_studies: synthetic correctness bench."""
    return _finite_blob(violin_qa_studies.bench_violin_qa_studies(seed))
