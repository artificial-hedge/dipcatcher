import pytest


@pytest.mark.parametrize(
    "name",
    [
        "pampas_deer_qa_studies",
        "marsh_deer_qa_studies",
        "tufted_qa_studies",
        "axis_qa_studies",
        "water_deer_qa_studies",
        "musk_deer_qa_studies",
    ],
)
def test_w1603_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
