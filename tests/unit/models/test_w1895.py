import pytest


@pytest.mark.parametrize(
    "name",
    [
        "gallu_qa_studies",
        "dimme_qa_studies",
        "lilu_qa_studies",
        "belili_qa_studies",
        "allatu_qa_studies",
        "sulak_qa_studies",
    ],
)
def test_w1895_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
