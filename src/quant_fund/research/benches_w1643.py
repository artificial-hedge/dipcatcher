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
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
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
