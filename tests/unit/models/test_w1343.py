import pytest


@pytest.mark.parametrize(
    "name",
    [
        "drop_qa_studies",
        "quoref_qa_studies",
        "news_qa_studies",
        "quail_qa_studies",
        "quac_qa_studies",
        "coqa_qa_studies",
    ],
)
def test_w1343_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
