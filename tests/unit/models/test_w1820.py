import pytest


@pytest.mark.parametrize(
    "name",
    [
        "yaoji2_qa_studies",
        "yandi2_qa_studies",
        "shennong2_qa_studies",
        "xihe2_qa_studies",
        "changyi2_qa_studies",
        "gun2_qa_studies",
    ],
)
def test_w1820_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
