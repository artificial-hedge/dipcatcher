import pytest


@pytest.mark.parametrize(
    "name",
    [
        "vilkacis_qa_studies",
        "lauma_qa_studies",
        "kaukas_qa_studies",
        "baubas_qa_studies",
        "pukis_qa_studies",
        "spigana_qa_studies",
    ],
)
def test_w1921_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
