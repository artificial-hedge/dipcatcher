import pytest

from quant_fund.research import benches_w1272


@pytest.mark.parametrize(
    "fam",
    [
        "bench_audio_encoder_studies_family",
        "bench_document_ai_studies_family",
        "bench_omni_modal_studies_family",
        "bench_unified_tokenizer_studies_family",
        "bench_video_llm_studies_family",
        "bench_visual_grounding_studies_family",
    ],
)
def test_benches_w1272(fam):
    out = getattr(benches_w1272, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
