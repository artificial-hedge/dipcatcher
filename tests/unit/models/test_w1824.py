import pytest


@pytest.mark.parametrize(
    "name",
    [
        "turul2_qa_studies",
        "remete2_qa_studies",
        "liderec2_qa_studies",
        "sarkany2_qa_studies",
        "tatros2_qa_studies",
        "boszorka2_qa_studies",
    ],
)
def test_w1824_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
