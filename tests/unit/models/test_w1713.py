import pytest


@pytest.mark.parametrize(
    "name",
    [
        "amirani_qa_studies",
        "ghmerti_qa_studies",
        "barbale_qa_studies",
        "apsat_qa_studies",
        "kamar_qa_studies",
        "dalis_qa_studies",
    ],
)
def test_w1713_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
