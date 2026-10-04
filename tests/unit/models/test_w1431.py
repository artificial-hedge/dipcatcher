import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bike_qa_studies",
        "car_qa_studies",
        "bus_qa_studies",
        "engine_qa_studies",
        "plane_qa_studies",
        "aircraft_qa_studies",
    ],
)
def test_w1431_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
