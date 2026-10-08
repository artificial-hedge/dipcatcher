"""Wave-1272 bench adapters: omni-modal canon (SYNTHETIC only)."""

from quant_fund.models import (
    audio_encoder_studies,
    document_ai_studies,
    omni_modal_studies,
    unified_tokenizer_studies,
    video_llm_studies,
    visual_grounding_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12720


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


def bench_audio_encoder_studies_family(seed: int = _SEED + 0):
    """audio_encoder_studies: synthetic correctness bench."""
    return _finite_blob(audio_encoder_studies.bench_audio_encoder_studies(seed))


def bench_document_ai_studies_family(seed: int = _SEED + 1):
    """document_ai_studies: synthetic correctness bench."""
    return _finite_blob(document_ai_studies.bench_document_ai_studies(seed))


def bench_omni_modal_studies_family(seed: int = _SEED + 2):
    """omni_modal_studies: synthetic correctness bench."""
    return _finite_blob(omni_modal_studies.bench_omni_modal_studies(seed))


def bench_unified_tokenizer_studies_family(seed: int = _SEED + 3):
    """unified_tokenizer_studies: synthetic correctness bench."""
    return _finite_blob(unified_tokenizer_studies.bench_unified_tokenizer_studies(seed))


def bench_video_llm_studies_family(seed: int = _SEED + 4):
    """video_llm_studies: synthetic correctness bench."""
    return _finite_blob(video_llm_studies.bench_video_llm_studies(seed))


def bench_visual_grounding_studies_family(seed: int = _SEED + 5):
    """visual_grounding_studies: synthetic correctness bench."""
    return _finite_blob(visual_grounding_studies.bench_visual_grounding_studies(seed))
