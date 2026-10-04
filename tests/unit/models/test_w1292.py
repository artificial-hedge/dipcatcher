import pytest


@pytest.mark.parametrize(
    "name",
    [
        "grpo_studies",
        "math_reward_studies",
        "outcome_reward_studies",
        "process_reward_studies",
        "rlvr_studies",
        "verifiable_reward_studies",
    ],
)
def test_w1292_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
