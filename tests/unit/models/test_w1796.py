import pytest


@pytest.mark.parametrize(
    "name",
    [
        "baldr2_qa_studies",
        "ullr2_qa_studies",
        "forseti2_qa_studies",
        "idun2_qa_studies",
        "hermodr_qa_studies",
        "nanna3_qa_studies",
    ],
)
def test_w1796_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
