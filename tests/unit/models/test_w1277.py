import pytest


@pytest.mark.parametrize(
    "name",
    [
        "debate_alignment_studies",
        "deliberative_alignment_studies",
        "iterated_amplification_studies",
        "recursive_reward_studies",
        "scalable_oversight_studies",
        "weak_to_strong_studies",
    ],
)
def test_w1277_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
