import pytest


@pytest.mark.parametrize(
    "name",
    [
        "oread_qa_studies",
        "napaea_qa_studies",
        "alseid_qa_studies",
        "meliae_qa_studies",
        "sylph_qa_studies",
        "gnome_volk_qa_studies",
    ],
)
def test_w1670_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
