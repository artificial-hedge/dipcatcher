import pytest


@pytest.mark.parametrize(
    "name",
    [
        "guzil_qa_studies",
        "warpon_qa_studies",
        "igal_qa_studies",
        "amayya_qa_studies",
        "tiniri_qa_studies",
        "atete_qa_studies",
    ],
)
def test_w1877_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
