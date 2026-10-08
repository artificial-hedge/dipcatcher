"""Wave-1380 bench adapters: intent-paraphrase canon (SYNTHETIC only)."""

from quant_fund.models import (
    art_nli_studies,
    para_paws_studies,
    recast_lite_studies,
    snips_lite_studies,
    social_lite_studies,
    subj_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13800


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


def bench_art_nli_studies_family(seed: int = _SEED + 0):
    """art_nli_studies: synthetic correctness bench."""
    return _finite_blob(art_nli_studies.bench_art_nli_studies(seed))


def bench_para_paws_studies_family(seed: int = _SEED + 1):
    """para_paws_studies: synthetic correctness bench."""
    return _finite_blob(para_paws_studies.bench_para_paws_studies(seed))


def bench_recast_lite_studies_family(seed: int = _SEED + 2):
    """recast_lite_studies: synthetic correctness bench."""
    return _finite_blob(recast_lite_studies.bench_recast_lite_studies(seed))


def bench_snips_lite_studies_family(seed: int = _SEED + 3):
    """snips_lite_studies: synthetic correctness bench."""
    return _finite_blob(snips_lite_studies.bench_snips_lite_studies(seed))


def bench_social_lite_studies_family(seed: int = _SEED + 4):
    """social_lite_studies: synthetic correctness bench."""
    return _finite_blob(social_lite_studies.bench_social_lite_studies(seed))


def bench_subj_lite_studies_family(seed: int = _SEED + 5):
    """subj_lite_studies: synthetic correctness bench."""
    return _finite_blob(subj_lite_studies.bench_subj_lite_studies(seed))
