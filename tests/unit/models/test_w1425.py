import pytest


@pytest.mark.parametrize(
    "name",
    [
        "broadcast_qa_studies",
        "debate_qa_studies",
        "column_qa_studies",
        "editorial_qa_studies",
        "headline_qa_studies",
        "article_qa_studies",
    ],
)
def test_w1425_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
