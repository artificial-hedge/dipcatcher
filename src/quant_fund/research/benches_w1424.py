"""Wave-1424 bench adapters: design-spec canon (SYNTHETIC only)."""

from quant_fund.models import (
    blueprint_qa_studies,
    design_qa_studies,
    format_qa_studies,
    layout_qa_studies,
    pattern_qa_studies,
    schema_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14240


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


def bench_blueprint_qa_studies_family(seed: int = _SEED + 0):
    """blueprint_qa_studies: synthetic correctness bench."""
    return _finite_blob(blueprint_qa_studies.bench_blueprint_qa_studies(seed))


def bench_design_qa_studies_family(seed: int = _SEED + 1):
    """design_qa_studies: synthetic correctness bench."""
    return _finite_blob(design_qa_studies.bench_design_qa_studies(seed))


def bench_format_qa_studies_family(seed: int = _SEED + 2):
    """format_qa_studies: synthetic correctness bench."""
    return _finite_blob(format_qa_studies.bench_format_qa_studies(seed))


def bench_layout_qa_studies_family(seed: int = _SEED + 3):
    """layout_qa_studies: synthetic correctness bench."""
    return _finite_blob(layout_qa_studies.bench_layout_qa_studies(seed))


def bench_pattern_qa_studies_family(seed: int = _SEED + 4):
    """pattern_qa_studies: synthetic correctness bench."""
    return _finite_blob(pattern_qa_studies.bench_pattern_qa_studies(seed))


def bench_schema_qa_studies_family(seed: int = _SEED + 5):
    """schema_qa_studies: synthetic correctness bench."""
    return _finite_blob(schema_qa_studies.bench_schema_qa_studies(seed))
