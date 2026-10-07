"""Wave-1358 bench adapters: reading-comp-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    adver_qa_studies,
    coqa_lite_studies,
    drop_lite_studies,
    duo_rc_studies,
    quac_lite_studies,
    trivia_web_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13580


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


def bench_adver_qa_studies_family(seed: int = _SEED + 0):
    """adver_qa_studies: synthetic correctness bench."""
    return _finite_blob(adver_qa_studies.bench_adver_qa_studies(seed))


def bench_coqa_lite_studies_family(seed: int = _SEED + 1):
    """coqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(coqa_lite_studies.bench_coqa_lite_studies(seed))


def bench_drop_lite_studies_family(seed: int = _SEED + 2):
    """drop_lite_studies: synthetic correctness bench."""
    return _finite_blob(drop_lite_studies.bench_drop_lite_studies(seed))


def bench_duo_rc_studies_family(seed: int = _SEED + 3):
    """duo_rc_studies: synthetic correctness bench."""
    return _finite_blob(duo_rc_studies.bench_duo_rc_studies(seed))


def bench_quac_lite_studies_family(seed: int = _SEED + 4):
    """quac_lite_studies: synthetic correctness bench."""
    return _finite_blob(quac_lite_studies.bench_quac_lite_studies(seed))


def bench_trivia_web_studies_family(seed: int = _SEED + 5):
    """trivia_web_studies: synthetic correctness bench."""
    return _finite_blob(trivia_web_studies.bench_trivia_web_studies(seed))
