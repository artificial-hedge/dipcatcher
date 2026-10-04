import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bary_score_studies",
        "kl_div_eval_studies",
        "gleu_lite_studies",
        "rouge_we_studies",
        "wmt_metric_studies",
        "cider_lite_studies",
    ],
)
def test_w1379_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
