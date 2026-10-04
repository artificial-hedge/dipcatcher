import pytest


@pytest.mark.parametrize(
    "name",
    [
        "humaneval_plus_studies",
        "mbpp_plus_studies",
        "swe_perf_studies",
        "livecodebench_studies",
        "bigcodebench_studies",
        "ds1000_studies",
    ],
)
def test_w1323_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
