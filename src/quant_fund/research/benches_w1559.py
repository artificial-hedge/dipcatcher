"""Wave-1559 bench adapters: pelagic-fish canon (SYNTHETIC only)."""

from quant_fund.models import (
    anchovy_qa_studies,
    bonito_qa_studies,
    herring_qa_studies,
    kingfish_qa_studies,
    mackerel_qa_studies,
    sardine_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15590


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_anchovy_qa_studies_family(seed: int = _SEED + 0):
    """anchovy_qa_studies: synthetic correctness bench."""
    return _finite_blob(anchovy_qa_studies.bench_anchovy_qa_studies(seed))


def bench_bonito_qa_studies_family(seed: int = _SEED + 1):
    """bonito_qa_studies: synthetic correctness bench."""
    return _finite_blob(bonito_qa_studies.bench_bonito_qa_studies(seed))


def bench_herring_qa_studies_family(seed: int = _SEED + 2):
    """herring_qa_studies: synthetic correctness bench."""
    return _finite_blob(herring_qa_studies.bench_herring_qa_studies(seed))


def bench_kingfish_qa_studies_family(seed: int = _SEED + 3):
    """kingfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(kingfish_qa_studies.bench_kingfish_qa_studies(seed))


def bench_mackerel_qa_studies_family(seed: int = _SEED + 4):
    """mackerel_qa_studies: synthetic correctness bench."""
    return _finite_blob(mackerel_qa_studies.bench_mackerel_qa_studies(seed))


def bench_sardine_qa_studies_family(seed: int = _SEED + 5):
    """sardine_qa_studies: synthetic correctness bench."""
    return _finite_blob(sardine_qa_studies.bench_sardine_qa_studies(seed))
