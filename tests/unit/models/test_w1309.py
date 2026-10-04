import pytest


@pytest.mark.parametrize(
    "name",
    [
        "gpqa_studies",
        "mmlu_pro_studies",
        "hle_studies",
        "frontier_math_studies",
        "workarena_studies",
        "tau_bench_studies",
    ],
)
def test_w1309_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
