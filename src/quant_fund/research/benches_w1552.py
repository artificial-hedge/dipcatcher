"""Wave-1552 bench adapters: turtle canon (SYNTHETIC only)."""

from quant_fund.models import (
    box_turtle_qa_studies,
    map_turtle_qa_studies,
    painted_turtle_qa_studies,
    slider_qa_studies,
    snapping_turtle_qa_studies,
    tortoise_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15520


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_box_turtle_qa_studies_family(seed: int = _SEED + 0):
    """box_turtle_qa_studies: synthetic correctness bench."""
    return _finite_blob(box_turtle_qa_studies.bench_box_turtle_qa_studies(seed))


def bench_map_turtle_qa_studies_family(seed: int = _SEED + 1):
    """map_turtle_qa_studies: synthetic correctness bench."""
    return _finite_blob(map_turtle_qa_studies.bench_map_turtle_qa_studies(seed))


def bench_painted_turtle_qa_studies_family(seed: int = _SEED + 2):
    """painted_turtle_qa_studies: synthetic correctness bench."""
    return _finite_blob(painted_turtle_qa_studies.bench_painted_turtle_qa_studies(seed))


def bench_slider_qa_studies_family(seed: int = _SEED + 3):
    """slider_qa_studies: synthetic correctness bench."""
    return _finite_blob(slider_qa_studies.bench_slider_qa_studies(seed))


def bench_snapping_turtle_qa_studies_family(seed: int = _SEED + 4):
    """snapping_turtle_qa_studies: synthetic correctness bench."""
    return _finite_blob(snapping_turtle_qa_studies.bench_snapping_turtle_qa_studies(seed))


def bench_tortoise_qa_studies_family(seed: int = _SEED + 5):
    """tortoise_qa_studies: synthetic correctness bench."""
    return _finite_blob(tortoise_qa_studies.bench_tortoise_qa_studies(seed))
