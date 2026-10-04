"""Wave-1643 bench adapters: greek-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    centaur_2_qa_studies,
    gryphon_qa_studies,
    harpy_2_qa_studies,
    hippogryph_qa_studies,
    minotaur_2_qa_studies,
    satyr_2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16430


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_centaur_2_qa_studies_family(seed: int = _SEED + 0):
    """centaur_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(centaur_2_qa_studies.bench_centaur_2_qa_studies(seed))


def bench_gryphon_qa_studies_family(seed: int = _SEED + 1):
    """gryphon_qa_studies: synthetic correctness bench."""
    return _finite_blob(gryphon_qa_studies.bench_gryphon_qa_studies(seed))


def bench_harpy_2_qa_studies_family(seed: int = _SEED + 2):
    """harpy_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(harpy_2_qa_studies.bench_harpy_2_qa_studies(seed))


def bench_hippogryph_qa_studies_family(seed: int = _SEED + 3):
    """hippogryph_qa_studies: synthetic correctness bench."""
    return _finite_blob(hippogryph_qa_studies.bench_hippogryph_qa_studies(seed))


def bench_minotaur_2_qa_studies_family(seed: int = _SEED + 4):
    """minotaur_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(minotaur_2_qa_studies.bench_minotaur_2_qa_studies(seed))


def bench_satyr_2_qa_studies_family(seed: int = _SEED + 5):
    """satyr_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(satyr_2_qa_studies.bench_satyr_2_qa_studies(seed))
