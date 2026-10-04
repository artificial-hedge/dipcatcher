"""Wave-1645 bench adapters: monster canon (SYNTHETIC only)."""

from quant_fund.models import (
    argus_qa_studies,
    cerberus_2_qa_studies,
    nemean_qa_studies,
    orthrus_qa_studies,
    pegasus_2_qa_studies,
    typhon_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16450


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_argus_qa_studies_family(seed: int = _SEED + 0):
    """argus_qa_studies: synthetic correctness bench."""
    return _finite_blob(argus_qa_studies.bench_argus_qa_studies(seed))


def bench_cerberus_2_qa_studies_family(seed: int = _SEED + 1):
    """cerberus_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(cerberus_2_qa_studies.bench_cerberus_2_qa_studies(seed))


def bench_nemean_qa_studies_family(seed: int = _SEED + 2):
    """nemean_qa_studies: synthetic correctness bench."""
    return _finite_blob(nemean_qa_studies.bench_nemean_qa_studies(seed))


def bench_orthrus_qa_studies_family(seed: int = _SEED + 3):
    """orthrus_qa_studies: synthetic correctness bench."""
    return _finite_blob(orthrus_qa_studies.bench_orthrus_qa_studies(seed))


def bench_pegasus_2_qa_studies_family(seed: int = _SEED + 4):
    """pegasus_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(pegasus_2_qa_studies.bench_pegasus_2_qa_studies(seed))


def bench_typhon_qa_studies_family(seed: int = _SEED + 5):
    """typhon_qa_studies: synthetic correctness bench."""
    return _finite_blob(typhon_qa_studies.bench_typhon_qa_studies(seed))
