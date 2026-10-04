import pytest


@pytest.mark.parametrize(
    "name",
    [
        "math_bench_studies",
        "wild_bench_studies",
        "multirc_studies",
        "objectnet_studies",
        "ood_bench_studies",
        "ninco_studies",
    ],
)
def test_w1319_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
