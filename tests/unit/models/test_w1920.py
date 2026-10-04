import pytest


@pytest.mark.parametrize(
    "name",
    [
        "strigoi_qa_studies",
        "moroi_qa_studies",
        "varcolac_qa_studies",
        "pricolici_qa_studies",
        "iele_qa_studies",
        "samca_qa_studies",
    ],
)
def test_w1920_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
