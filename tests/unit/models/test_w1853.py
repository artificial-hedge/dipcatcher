import pytest


@pytest.mark.parametrize(
    "name",
    [
        "apedemak_qa_studies",
        "amesemi_qa_studies",
        "sebiumeker_qa_studies",
        "dedun_qa_studies",
        "sabios_qa_studies",
        "aresnuphis_qa_studies",
    ],
)
def test_w1853_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
