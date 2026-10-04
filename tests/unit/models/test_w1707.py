import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ukko_qa_studies",
        "tapio_qa_studies",
        "ahti_qa_studies",
        "ilmatar_qa_studies",
        "louhi_qa_studies",
        "kiputytto_qa_studies",
    ],
)
def test_w1707_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
