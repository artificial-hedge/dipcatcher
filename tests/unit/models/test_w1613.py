import pytest


@pytest.mark.parametrize(
    "name",
    [
        "saola_qa_studies",
        "yak_qa_studies",
        "aurochs_qa_studies",
        "tamaraw_qa_studies",
        "gaur_qa_studies",
        "banteng_qa_studies",
    ],
)
def test_w1613_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
