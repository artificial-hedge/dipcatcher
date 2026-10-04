import pytest


@pytest.mark.parametrize(
    "name",
    [
        "harm_bench_studies",
        "jailbreak_bench_studies",
        "prompt_inject_studies",
        "agent_harm_studies",
        "safety_bench_studies",
        "xstest_studies",
    ],
)
def test_w1299_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
