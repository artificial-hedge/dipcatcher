import pytest


@pytest.mark.parametrize(
    "name",
    [
        "chart_qa_lite_studies",
        "infovqa_lite_studies",
        "docvqa_lite_studies",
        "mmqa_lite_studies",
        "ocrvqa_lite_studies",
        "ai2d_lite_studies",
    ],
)
def test_w1403_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
