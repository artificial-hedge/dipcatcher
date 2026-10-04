import pytest


@pytest.mark.parametrize(
    "name",
    [
        "capability_eval_studies",
        "control_eval_studies",
        "deception_eval_studies",
        "prompt_injection_studies",
        "sandbox_escape_studies",
        "tool_call_verify_studies",
    ],
)
def test_w1287_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
