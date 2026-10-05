import pytest


@pytest.mark.parametrize(
    "name",
    [
        "laime2_qa_studies",
        "aitvaras2_qa_studies",
        "velnias2_qa_studies",
        "zemyna2_qa_studies",
        "perkunas2_qa_studies",
        "kaukas2_qa_studies",
    ],
)
def test_w1825_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
