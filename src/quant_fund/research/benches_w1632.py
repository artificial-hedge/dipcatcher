"""Wave-1632 bench adapters: legendary-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    cerberus_qa_studies,
    dragon_2_qa_studies,
    minotaur_qa_studies,
    pegasus_qa_studies,
    phoenix_2_qa_studies,
    unicorn_2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16320


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cerberus_qa_studies_family(seed: int = _SEED + 0):
    """cerberus_qa_studies: synthetic correctness bench."""
    return _finite_blob(cerberus_qa_studies.bench_cerberus_qa_studies(seed))


def bench_dragon_2_qa_studies_family(seed: int = _SEED + 1):
    """dragon_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(dragon_2_qa_studies.bench_dragon_2_qa_studies(seed))


def bench_minotaur_qa_studies_family(seed: int = _SEED + 2):
    """minotaur_qa_studies: synthetic correctness bench."""
    return _finite_blob(minotaur_qa_studies.bench_minotaur_qa_studies(seed))


def bench_pegasus_qa_studies_family(seed: int = _SEED + 3):
    """pegasus_qa_studies: synthetic correctness bench."""
    return _finite_blob(pegasus_qa_studies.bench_pegasus_qa_studies(seed))


def bench_phoenix_2_qa_studies_family(seed: int = _SEED + 4):
    """phoenix_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(phoenix_2_qa_studies.bench_phoenix_2_qa_studies(seed))


def bench_unicorn_2_qa_studies_family(seed: int = _SEED + 5):
    """unicorn_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(unicorn_2_qa_studies.bench_unicorn_2_qa_studies(seed))
