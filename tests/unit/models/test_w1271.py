import pytest


@pytest.mark.parametrize(
    "name",
    [
        "arena_battle_studies",
        "bigbench_studies",
        "capability_elicitation_studies",
        "contamination_detect_studies",
        "helm_eval_studies",
        "llm_judge_studies",
    ],
)
def test_w1271_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
