import pytest


@pytest.mark.parametrize(
    "name",
    [
        "blender_bot_studies",
        "dstc_lite_studies",
        "daily_dialog_studies",
        "empathy_dialog_studies",
        "persona_chat_studies",
        "conv_ai2_studies",
    ],
)
def test_w1376_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
