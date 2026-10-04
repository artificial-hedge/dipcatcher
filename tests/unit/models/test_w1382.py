import pytest


@pytest.mark.parametrize(
    "name",
    [
        "fact_score_studies",
        "lmsys_eval_studies",
        "helm_lite_studies",
        "nugget_eval_studies",
        "vicuna_bench_studies",
        "gpt_score_studies",
    ],
)
def test_w1382_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
