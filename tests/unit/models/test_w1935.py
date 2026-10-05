import pytest


@pytest.mark.parametrize(
    "name",
    [
        "jarjacha_qa_studies",
        "anchancho_qa_studies",
        "muki_qa_studies",
        "pishtaco_qa_studies",
        "kharisiri_qa_studies",
        "sirenito_qa_studies",
    ],
)
def test_w1935_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
