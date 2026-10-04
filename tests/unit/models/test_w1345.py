import pytest


@pytest.mark.parametrize(
    "name",
    [
        "trivia_qa_studies",
        "nq_open_studies",
        "web_qa_studies",
        "freebase_qa_studies",
        "complex_qa_studies",
        "entity_quests_studies",
    ],
)
def test_w1345_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
