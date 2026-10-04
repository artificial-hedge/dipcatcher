import pytest


@pytest.mark.parametrize(
    "name",
    [
        "fallow_qa_studies",
        "roe_qa_studies",
        "chital_qa_studies",
        "sika_qa_studies",
        "muntjac_qa_studies",
        "pudu_qa_studies",
    ],
)
def test_w1580_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
