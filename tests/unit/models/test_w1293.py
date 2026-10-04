import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cai_critique_studies",
        "constitutional_studies",
        "harmlessness_rl_studies",
        "principle_eval_studies",
        "rlaif_studies",
        "sleeper_eval_studies",
    ],
)
def test_w1293_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
