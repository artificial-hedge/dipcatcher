import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ecare_lite_studies",
        "event_qa_studies",
        "event2mind_lite_studies",
        "hippo_qa_studies",
        "intent_qa_studies",
        "causal_qa_studies",
    ],
)
def test_w1410_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
