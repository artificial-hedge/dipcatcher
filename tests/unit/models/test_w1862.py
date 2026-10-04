import pytest


@pytest.mark.parametrize(
    "name",
    [
        "isolde_qa_studies",
        "mark_cornwall_qa_studies",
        "lyonesse_qa_studies",
        "mordred_qa_studies",
        "agravaine_qa_studies",
        "kay_qa_studies",
    ],
)
def test_w1862_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
