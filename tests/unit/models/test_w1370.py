import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bbh_lite_studies",
        "live_bench_studies",
        "if_eval_studies",
        "olympic_bench_studies",
        "trivia_qa_lite_studies",
        "gpqa_lite_studies",
    ],
)
def test_w1370_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
