import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ptarmigan_qa_studies",
        "arctic_hare_qa_studies",
        "snowshoe_qa_studies",
        "tundra_swan_qa_studies",
        "gyrfalcon_qa_studies",
        "pallas_manul_qa_studies",
    ],
)
def test_w1624_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
