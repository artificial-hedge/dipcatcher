"""Wave-1581 bench adapters: pinniped canon (SYNTHETIC only)."""

from quant_fund.models import (
    elephant_seal_qa_studies,
    fur_seal_qa_studies,
    harp_seal_qa_studies,
    leopard_seal_qa_studies,
    monk_seal_qa_studies,
    weddell_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15810


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_elephant_seal_qa_studies_family(seed: int = _SEED + 0):
    """elephant_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(elephant_seal_qa_studies.bench_elephant_seal_qa_studies(seed))


def bench_fur_seal_qa_studies_family(seed: int = _SEED + 1):
    """fur_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(fur_seal_qa_studies.bench_fur_seal_qa_studies(seed))


def bench_harp_seal_qa_studies_family(seed: int = _SEED + 2):
    """harp_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(harp_seal_qa_studies.bench_harp_seal_qa_studies(seed))


def bench_leopard_seal_qa_studies_family(seed: int = _SEED + 3):
    """leopard_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(leopard_seal_qa_studies.bench_leopard_seal_qa_studies(seed))


def bench_monk_seal_qa_studies_family(seed: int = _SEED + 4):
    """monk_seal_qa_studies: synthetic correctness bench."""
    return _finite_blob(monk_seal_qa_studies.bench_monk_seal_qa_studies(seed))


def bench_weddell_qa_studies_family(seed: int = _SEED + 5):
    """weddell_qa_studies: synthetic correctness bench."""
    return _finite_blob(weddell_qa_studies.bench_weddell_qa_studies(seed))
