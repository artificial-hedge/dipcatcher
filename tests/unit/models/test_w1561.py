import pytest


@pytest.mark.parametrize(
    "name",
    [
        "surgeonfish_qa_studies",
        "lionfish_qa_studies",
        "blenny_qa_studies",
        "triggerfish_qa_studies",
        "angelfish_qa_studies",
        "goby_qa_studies",
    ],
)
def test_w1561_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
