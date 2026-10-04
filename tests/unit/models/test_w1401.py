import pytest


@pytest.mark.parametrize(
    "name",
    [
        "dolphin_lite_studies",
        "lila_lite_studies",
        "draw_lite_studies",
        "math_doc_studies",
        "math_eval_studies",
        "alg514_lite_studies",
    ],
)
def test_w1401_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
