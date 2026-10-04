"""Wave-1460 bench adapters: savanna canon (SYNTHETIC only)."""

from quant_fund.models import (
    baboon_qa_studies,
    elephant_qa_studies,
    gazelle_qa_studies,
    giraffe_qa_studies,
    wildebeest_qa_studies,
    zebra_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14600


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_baboon_qa_studies_family(seed: int = _SEED + 0):
    """baboon_qa_studies: synthetic correctness bench."""
    return _finite_blob(baboon_qa_studies.bench_baboon_qa_studies(seed))


def bench_elephant_qa_studies_family(seed: int = _SEED + 1):
    """elephant_qa_studies: synthetic correctness bench."""
    return _finite_blob(elephant_qa_studies.bench_elephant_qa_studies(seed))


def bench_gazelle_qa_studies_family(seed: int = _SEED + 2):
    """gazelle_qa_studies: synthetic correctness bench."""
    return _finite_blob(gazelle_qa_studies.bench_gazelle_qa_studies(seed))


def bench_giraffe_qa_studies_family(seed: int = _SEED + 3):
    """giraffe_qa_studies: synthetic correctness bench."""
    return _finite_blob(giraffe_qa_studies.bench_giraffe_qa_studies(seed))


def bench_wildebeest_qa_studies_family(seed: int = _SEED + 4):
    """wildebeest_qa_studies: synthetic correctness bench."""
    return _finite_blob(wildebeest_qa_studies.bench_wildebeest_qa_studies(seed))


def bench_zebra_qa_studies_family(seed: int = _SEED + 5):
    """zebra_qa_studies: synthetic correctness bench."""
    return _finite_blob(zebra_qa_studies.bench_zebra_qa_studies(seed))
