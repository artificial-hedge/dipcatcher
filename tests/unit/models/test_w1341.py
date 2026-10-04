import pytest


@pytest.mark.parametrize(
    "name",
    [
        "crow_s_pairs_studies",
        "stereo_set_studies",
        "real_toxicity_studies",
        "bold_eval_studies",
        "hate_speech_eval_studies",
        "holo_bias_studies",
    ],
)
def test_w1341_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
