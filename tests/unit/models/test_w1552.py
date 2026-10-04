import pytest


@pytest.mark.parametrize(
    "name",
    [
        "painted_turtle_qa_studies",
        "map_turtle_qa_studies",
        "slider_qa_studies",
        "snapping_turtle_qa_studies",
        "box_turtle_qa_studies",
        "tortoise_qa_studies",
    ],
)
def test_w1552_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
