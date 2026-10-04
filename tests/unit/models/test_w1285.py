import pytest


@pytest.mark.parametrize(
    "name",
    [
        "audio_lm_studies",
        "chart_reasoning_studies",
        "doc_vqa_studies",
        "gui_agent_studies",
        "video_understanding_studies",
        "vision_pretraining_studies",
    ],
)
def test_w1285_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
