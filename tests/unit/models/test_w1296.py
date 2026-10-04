import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ensemble_rm_studies",
        "judge_reward_studies",
        "margin_reward_studies",
        "reward_hacking_studies",
        "reward_uncertainty_studies",
        "rm_btd_studies",
    ],
)
def test_w1296_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
