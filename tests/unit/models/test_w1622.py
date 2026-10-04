import pytest


@pytest.mark.parametrize(
    "name",
    [
        "blue_sheep_qa_studies",
        "barbary_qa_studies",
        "himalayan_qa_studies",
        "snow_leopard_qa_studies",
        "snowcock_qa_studies",
        "nilgiri_qa_studies",
    ],
)
def test_w1622_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
