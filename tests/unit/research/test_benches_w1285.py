import pytest

from quant_fund.research import benches_w1285


@pytest.mark.parametrize(
    "fam",
    [
        "bench_audio_lm_studies_family",
        "bench_chart_reasoning_studies_family",
        "bench_doc_vqa_studies_family",
        "bench_gui_agent_studies_family",
        "bench_video_understanding_studies_family",
        "bench_vision_pretraining_studies_family",
    ],
)
def test_benches_w1285(fam):
    out = getattr(benches_w1285, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
