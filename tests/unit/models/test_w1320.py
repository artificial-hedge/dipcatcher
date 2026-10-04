import pytest


@pytest.mark.parametrize(
    "name",
    [
        "osworld_studies",
        "webvoyager_studies",
        "gaia_bench_studies",
        "mmbench_agent_studies",
        "screen_eval_studies",
        "vsi_bench_studies",
    ],
)
def test_w1320_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
