import pytest


@pytest.mark.parametrize(
    "name",
    [
        "pseudoscorpion_qa_studies",
        "solifuge_qa_studies",
        "tick_qa_studies",
        "whip_scorpion_qa_studies",
        "harvestman_qa_studies",
        "vinegaroon_qa_studies",
    ],
)
def test_w1555_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
