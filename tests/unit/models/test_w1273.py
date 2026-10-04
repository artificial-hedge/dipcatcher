import pytest


@pytest.mark.parametrize(
    "name",
    [
        "alignment_eval_studies",
        "guardrail_studies",
        "hallucination_detect_studies",
        "jailbreak_defense_studies",
        "red_team_studies",
        "sleeper_agent_studies",
    ],
)
def test_w1273_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
