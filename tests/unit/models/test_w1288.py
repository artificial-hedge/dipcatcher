import pytest


@pytest.mark.parametrize(
    "name",
    [
        "adversarial_irl_studies",
        "behavior_cloning_studies",
        "dagger_studies",
        "offline_distill_studies",
        "preference_irl_studies",
        "skill_extraction_studies",
    ],
)
def test_w1288_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
