import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hawker_qa_studies",
        "clubtail_qa_studies",
        "darner_qa_studies",
        "spreadwing_qa_studies",
        "forktail_qa_studies",
        "damselfly_qa_studies",
    ],
)
def test_w1517_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
