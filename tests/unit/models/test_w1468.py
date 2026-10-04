import pytest


@pytest.mark.parametrize(
    "name",
    [
        "fen_qa_studies",
        "heath_qa_studies",
        "glen_qa_studies",
        "knoll_qa_studies",
        "moor_qa_studies",
        "dale_qa_studies",
    ],
)
def test_w1468_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
