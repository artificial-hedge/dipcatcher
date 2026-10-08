"""Wave-1285 bench adapters: multimodal-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    audio_lm_studies,
    chart_reasoning_studies,
    doc_vqa_studies,
    gui_agent_studies,
    video_understanding_studies,
    vision_pretraining_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12850


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


def bench_audio_lm_studies_family(seed: int = _SEED + 0):
    """audio_lm_studies: synthetic correctness bench."""
    return _finite_blob(audio_lm_studies.bench_audio_lm_studies(seed))


def bench_chart_reasoning_studies_family(seed: int = _SEED + 1):
    """chart_reasoning_studies: synthetic correctness bench."""
    return _finite_blob(chart_reasoning_studies.bench_chart_reasoning_studies(seed))


def bench_doc_vqa_studies_family(seed: int = _SEED + 2):
    """doc_vqa_studies: synthetic correctness bench."""
    return _finite_blob(doc_vqa_studies.bench_doc_vqa_studies(seed))


def bench_gui_agent_studies_family(seed: int = _SEED + 3):
    """gui_agent_studies: synthetic correctness bench."""
    return _finite_blob(gui_agent_studies.bench_gui_agent_studies(seed))


def bench_video_understanding_studies_family(seed: int = _SEED + 4):
    """video_understanding_studies: synthetic correctness bench."""
    return _finite_blob(video_understanding_studies.bench_video_understanding_studies(seed))


def bench_vision_pretraining_studies_family(seed: int = _SEED + 5):
    """vision_pretraining_studies: synthetic correctness bench."""
    return _finite_blob(vision_pretraining_studies.bench_vision_pretraining_studies(seed))
