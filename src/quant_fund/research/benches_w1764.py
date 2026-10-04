"""Wave-1764 bench adapters: greek-myth-6 canon (SYNTHETIC only)."""

from quant_fund.models import (
    eileithyia_qa_studies,
    iris_qa_studies,
    leto_qa_studies,
    nemesis_qa_studies,
    nike_qa_studies,
    tyche_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17640


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_eileithyia_qa_studies_family(seed: int = _SEED + 0):
    """eileithyia_qa_studies: synthetic correctness bench."""
    return _finite_blob(eileithyia_qa_studies.bench_eileithyia_qa_studies(seed))


def bench_iris_qa_studies_family(seed: int = _SEED + 1):
    """iris_qa_studies: synthetic correctness bench."""
    return _finite_blob(iris_qa_studies.bench_iris_qa_studies(seed))


def bench_leto_qa_studies_family(seed: int = _SEED + 2):
    """leto_qa_studies: synthetic correctness bench."""
    return _finite_blob(leto_qa_studies.bench_leto_qa_studies(seed))


def bench_nemesis_qa_studies_family(seed: int = _SEED + 3):
    """nemesis_qa_studies: synthetic correctness bench."""
    return _finite_blob(nemesis_qa_studies.bench_nemesis_qa_studies(seed))


def bench_nike_qa_studies_family(seed: int = _SEED + 4):
    """nike_qa_studies: synthetic correctness bench."""
    return _finite_blob(nike_qa_studies.bench_nike_qa_studies(seed))


def bench_tyche_qa_studies_family(seed: int = _SEED + 5):
    """tyche_qa_studies: synthetic correctness bench."""
    return _finite_blob(tyche_qa_studies.bench_tyche_qa_studies(seed))
