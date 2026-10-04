import pytest


@pytest.mark.parametrize(
    "name",
    [
        "suzaku_qa_studies",
        "seiryu_qa_studies",
        "byakko_qa_studies",
        "genbu_qa_studies",
        "kohryu_qa_studies",
        "kirin_2_qa_studies",
    ],
)
def test_w1640_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
