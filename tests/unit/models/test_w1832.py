import pytest


@pytest.mark.parametrize(
    "name",
    [
        "melqart2_qa_studies",
        "eshmun2_qa_studies",
        "reshef2_qa_studies",
        "tanit2_qa_studies",
        "baalat2_qa_studies",
        "yam2_qa_studies",
    ],
)
def test_w1832_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
