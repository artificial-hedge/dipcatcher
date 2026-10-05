import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hathor2_qa_studies",
        "bastet2_qa_studies",
        "sekhmet2_qa_studies",
        "nut2_qa_studies",
        "geb2_qa_studies",
        "tefnut2_qa_studies",
    ],
)
def test_w1818_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
