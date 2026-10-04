"""Wave-1418 bench adapters: lore-reference canon (SYNTHETIC only)."""

from quant_fund.models import (
    almanac_qa_studies,
    atlas_qa_studies,
    idiom_qa_studies,
    jeopardy_qa_studies,
    misc_qa_studies,
    myth_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14180


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_almanac_qa_studies_family(seed: int = _SEED + 0):
    """almanac_qa_studies: synthetic correctness bench."""
    return _finite_blob(almanac_qa_studies.bench_almanac_qa_studies(seed))


def bench_atlas_qa_studies_family(seed: int = _SEED + 1):
    """atlas_qa_studies: synthetic correctness bench."""
    return _finite_blob(atlas_qa_studies.bench_atlas_qa_studies(seed))


def bench_idiom_qa_studies_family(seed: int = _SEED + 2):
    """idiom_qa_studies: synthetic correctness bench."""
    return _finite_blob(idiom_qa_studies.bench_idiom_qa_studies(seed))


def bench_jeopardy_qa_studies_family(seed: int = _SEED + 3):
    """jeopardy_qa_studies: synthetic correctness bench."""
    return _finite_blob(jeopardy_qa_studies.bench_jeopardy_qa_studies(seed))


def bench_misc_qa_studies_family(seed: int = _SEED + 4):
    """misc_qa_studies: synthetic correctness bench."""
    return _finite_blob(misc_qa_studies.bench_misc_qa_studies(seed))


def bench_myth_qa_studies_family(seed: int = _SEED + 5):
    """myth_qa_studies: synthetic correctness bench."""
    return _finite_blob(myth_qa_studies.bench_myth_qa_studies(seed))
