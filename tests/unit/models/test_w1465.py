import pytest


@pytest.mark.parametrize(
    "name",
    [
        "forge_qa_studies",
        "lighthouse_qa_studies",
        "grotto_qa_studies",
        "quarry_qa_studies",
        "vault_qa_studies",
        "citadel_qa_studies",
    ],
)
def test_w1465_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
