import pytest


@pytest.mark.parametrize(
    "name",
    [
        "tool_use_eval_studies",
        "browse_eval_studies",
        "swe_bench_studies",
        "terminal_bench_studies",
        "os_world_studies",
        "web_arena_studies",
    ],
)
def test_w1298_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
