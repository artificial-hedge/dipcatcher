import pytest


@pytest.mark.parametrize(
    "name",
    [
        "kratti_qa_studies",
        "nakki_qa_studies",
        "tursas_qa_studies",
        "kalma_qa_studies",
        "painajainen_qa_studies",
        "loviatar_qa_studies",
    ],
)
def test_w1922_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
