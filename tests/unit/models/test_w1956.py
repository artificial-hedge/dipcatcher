import pytest


@pytest.mark.parametrize(
    "name",
    [
        "halphas_qa_studies",
        "raum_qa_studies",
        "focalor_qa_studies",
        "vepar_qa_studies",
        "sabnock_qa_studies",
        "shax_qa_studies",
    ],
)
def test_w1956_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
