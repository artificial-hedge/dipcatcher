import pytest


@pytest.mark.parametrize(
    "name",
    [
        "fire_spirit_qa_studies",
        "water_sprite_qa_studies",
        "earth_golem_qa_studies",
        "air_sylph_qa_studies",
        "storm_jinn_qa_studies",
        "frost_wight_qa_studies",
    ],
)
def test_w1633_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
