import pytest


@pytest.mark.parametrize(
    "name",
    [
        "melqart_qa_studies",
        "eshmun_qa_studies",
        "baalat_qa_studies",
        "tanit_qa_studies",
        "dagon_qa_studies",
        "resheph_qa_studies",
    ],
)
def test_w1711_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
