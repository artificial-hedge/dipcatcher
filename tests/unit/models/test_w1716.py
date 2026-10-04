import pytest


@pytest.mark.parametrize(
    "name",
    [
        "kotys_qa_studies",
        "kottiso_qa_studies",
        "semele_qa_studies",
        "zibelthiurdos_qa_studies",
        "theandrites_qa_studies",
        "heroas_qa_studies",
    ],
)
def test_w1716_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
