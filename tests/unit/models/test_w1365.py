import pytest


@pytest.mark.parametrize(
    "name",
    [
        "arxiv_sum_studies",
        "multi_news_studies",
        "dialogsum_lite_studies",
        "pubmed_sum_studies",
        "samsum_lite_studies",
        "cnn_dailymail_studies",
    ],
)
def test_w1365_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
