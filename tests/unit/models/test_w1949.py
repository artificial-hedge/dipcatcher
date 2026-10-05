import pytest


@pytest.mark.parametrize(
    "name",
    [
        "gusion_qa_studies",
        "barbatos_qa_studies",
        "agares_qa_studies",
        "amon_qa_studies",
        "valefor_qa_studies",
        "vasago_qa_studies",
    ],
)
def test_w1949_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
