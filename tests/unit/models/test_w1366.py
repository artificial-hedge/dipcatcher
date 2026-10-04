import pytest


@pytest.mark.parametrize(
    "name",
    [
        "align_score_studies",
        "faith_eval_studies",
        "factcc_lite_studies",
        "quest_eval_studies",
        "summa_eval_studies",
        "dice_eval_studies",
    ],
)
def test_w1366_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
