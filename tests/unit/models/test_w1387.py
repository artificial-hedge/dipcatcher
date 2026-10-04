import pytest


@pytest.mark.parametrize(
    "name",
    [
        "api_blend_studies",
        "gta_bench_studies",
        "gorilla_eval_studies",
        "seal_tools_studies",
        "stabletoolbench_studies",
        "bfcl_v3_studies",
    ],
)
def test_w1387_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
