import pytest


@pytest.mark.parametrize(
    "name",
    [
        "num2_qa_studies",
        "naa2_qa_studies",
        "khosun2_qa_studies",
        "kyys2_qa_studies",
        "abaasy2_qa_studies",
        "buga2_qa_studies",
    ],
)
def test_w1827_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
