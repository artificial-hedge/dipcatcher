import pytest


@pytest.mark.parametrize(
    "name",
    [
        "eagle_ray_qa_studies",
        "sawfish_qa_studies",
        "torpedo_ray_qa_studies",
        "manta_qa_studies",
        "guitarfish_qa_studies",
        "thornback_qa_studies",
    ],
)
def test_w1569_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
