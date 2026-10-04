import pytest


@pytest.mark.parametrize(
    "name",
    [
        "analogical_prompt_studies",
        "cot_studies",
        "reflexion_studies",
        "scratchpad_studies",
        "self_consistency_studies",
        "stepwise_verify_studies",
    ],
)
def test_w1289_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
