import pytest


@pytest.mark.parametrize(
    "name",
    [
        "wmdp_studies",
        "cyber_sec_eval_studies",
        "bio_risk_eval_studies",
        "chem_risk_eval_studies",
        "lab_bench_studies",
        "malicious_instruct_studies",
    ],
)
def test_w1312_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
