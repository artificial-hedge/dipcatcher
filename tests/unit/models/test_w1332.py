import pytest


@pytest.mark.parametrize(
    "name",
    [
        "decontaminate_studies",
        "ngram_overlap_studies",
        "eval_bias_studies",
        "fair_eval_studies",
        "g_eval_studies",
        "pandalm_studies",
    ],
)
def test_w1332_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
