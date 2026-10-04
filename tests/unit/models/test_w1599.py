import pytest


@pytest.mark.parametrize(
    "name",
    [
        "howler_qa_studies",
        "mouse_lemur_qa_studies",
        "ring_tailed_qa_studies",
        "night_monkey_qa_studies",
        "aye_aye_qa_studies",
        "spider_monkey_qa_studies",
    ],
)
def test_w1599_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
