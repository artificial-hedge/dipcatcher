"""Wave-1375 bench adapters: scientific-summarization canon (SYNTHETIC only)."""

from quant_fund.models import (
    facet_sum_studies,
    ms2_lite_studies,
    patent_sum_studies,
    sci_lay_studies,
    scitldr_lite_studies,
    spectrum_sum_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13750


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


def bench_facet_sum_studies_family(seed: int = _SEED + 0):
    """facet_sum_studies: synthetic correctness bench."""
    return _finite_blob(facet_sum_studies.bench_facet_sum_studies(seed))


def bench_ms2_lite_studies_family(seed: int = _SEED + 1):
    """ms2_lite_studies: synthetic correctness bench."""
    return _finite_blob(ms2_lite_studies.bench_ms2_lite_studies(seed))


def bench_patent_sum_studies_family(seed: int = _SEED + 2):
    """patent_sum_studies: synthetic correctness bench."""
    return _finite_blob(patent_sum_studies.bench_patent_sum_studies(seed))


def bench_sci_lay_studies_family(seed: int = _SEED + 3):
    """sci_lay_studies: synthetic correctness bench."""
    return _finite_blob(sci_lay_studies.bench_sci_lay_studies(seed))


def bench_scitldr_lite_studies_family(seed: int = _SEED + 4):
    """scitldr_lite_studies: synthetic correctness bench."""
    return _finite_blob(scitldr_lite_studies.bench_scitldr_lite_studies(seed))


def bench_spectrum_sum_studies_family(seed: int = _SEED + 5):
    """spectrum_sum_studies: synthetic correctness bench."""
    return _finite_blob(spectrum_sum_studies.bench_spectrum_sum_studies(seed))
