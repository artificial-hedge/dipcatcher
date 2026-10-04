import pytest


@pytest.mark.parametrize(
    "name",
    [
        "swe_bench_verified_studies",
        "crosscodeeval_studies",
        "mer_bench_studies",
        "codescope_studies",
        "project_eval_studies",
        "concode_eval_studies",
    ],
)
def test_w1331_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
