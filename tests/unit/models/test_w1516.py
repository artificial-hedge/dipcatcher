import pytest


@pytest.mark.parametrize(
    "name",
    [
        "atlas_moth_qa_studies",
        "tussock_moth_qa_studies",
        "hawk_moth_qa_studies",
        "gypsy_moth_qa_studies",
        "underwing_qa_studies",
        "luna_moth_qa_studies",
    ],
)
def test_w1516_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
