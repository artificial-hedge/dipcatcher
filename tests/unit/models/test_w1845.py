import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hubal_qa_studies",
        "manat_qa_studies",
        "allat_qa_studies",
        "uzza_qa_studies",
        "wadd_qa_studies",
        "dushara_qa_studies",
    ],
)
def test_w1845_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
