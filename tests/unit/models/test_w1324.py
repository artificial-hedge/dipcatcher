import pytest


@pytest.mark.parametrize(
    "name",
    [
        "reward_bench_studies",
        "arena_hard_studies",
        "mt_bench_judge_studies",
        "alpacaeval_studies",
        "prometheus_eval_studies",
        "judge_bench_studies",
    ],
)
def test_w1324_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
