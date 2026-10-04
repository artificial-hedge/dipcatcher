import pytest


@pytest.mark.parametrize(
    "name",
    [
        "vampire_qa_studies",
        "horseshoe_bat_qa_studies",
        "leaf_nosed_qa_studies",
        "flying_fox_qa_studies",
        "pipistrelle_qa_studies",
        "noctule_qa_studies",
    ],
)
def test_w1582_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
