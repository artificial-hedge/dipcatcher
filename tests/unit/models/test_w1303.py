import pytest


@pytest.mark.parametrize(
    "name",
    [
        "mmlu_studies",
        "bbh_studies",
        "gsm8k_studies",
        "humaneval_studies",
        "mt_bench_studies",
        "ifeval_studies",
    ],
)
def test_w1303_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
