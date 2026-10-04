import pytest


@pytest.mark.parametrize(
    "name",
    [
        "coherence_qa_studies",
        "discourse_qa_studies",
        "dialogue_act_studies",
        "hedge_qa_studies",
        "implicit_qa_studies",
        "anaphora_qa_studies",
    ],
)
def test_w1413_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
