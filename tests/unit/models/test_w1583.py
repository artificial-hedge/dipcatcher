import pytest


@pytest.mark.parametrize(
    "name",
    [
        "fin_whale_qa_studies",
        "sperm_whale_qa_studies",
        "bowhead_qa_studies",
        "humpback_qa_studies",
        "pilot_whale_qa_studies",
        "minke_qa_studies",
    ],
)
def test_w1583_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
