import pytest


@pytest.mark.parametrize(
    "name",
    [
        "audio_encoder_studies",
        "document_ai_studies",
        "omni_modal_studies",
        "unified_tokenizer_studies",
        "video_llm_studies",
        "visual_grounding_studies",
    ],
)
def test_w1272_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
