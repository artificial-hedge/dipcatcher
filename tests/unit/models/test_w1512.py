import pytest


@pytest.mark.parametrize(
    "name",
    [
        "knot_qa_studies",
        "oystercatcher_qa_studies",
        "phalarope_qa_studies",
        "stilt_qa_studies",
        "whimbrel_qa_studies",
        "dunlin_qa_studies",
    ],
)
def test_w1512_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
