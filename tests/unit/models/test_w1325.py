import pytest


@pytest.mark.parametrize(
    "name",
    [
        "infinitebench_studies",
        "ruler_bench_studies",
        "longbench_studies",
        "babilong_studies",
        "zero_scrolls_studies",
        "lv_eval_studies",
    ],
)
def test_w1325_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
