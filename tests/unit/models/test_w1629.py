import pytest


@pytest.mark.parametrize(
    "name",
    [
        "thunderbird_qa_studies",
        "chupacabra_qa_studies",
        "mothman_qa_studies",
        "jersey_devil_qa_studies",
        "kraken_2_qa_studies",
        "yeti_2_qa_studies",
    ],
)
def test_w1629_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
