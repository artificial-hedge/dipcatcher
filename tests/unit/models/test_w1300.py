import pytest


@pytest.mark.parametrize(
    "name",
    [
        "corruption_studies",
        "imagenet_c_studies",
        "imagenet_r_studies",
        "adversarial_eval_studies",
        "autoattack_studies",
        "robust_bench_studies",
    ],
)
def test_w1300_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
