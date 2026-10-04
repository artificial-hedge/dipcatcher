import pytest


@pytest.mark.parametrize(
    "name",
    [
        "abduction_lite_studies",
        "dream_lite_studies",
        "conv_finqa_studies",
        "equiv_lite_studies",
        "wsc_lite_studies",
        "board_game_qa_studies",
    ],
)
def test_w1377_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
