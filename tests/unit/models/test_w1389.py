import pytest


@pytest.mark.parametrize(
    "name",
    [
        "book_sum_studies",
        "marlense_studies",
        "infinitesum_studies",
        "narra_sum_studies",
        "quote_sum_studies",
        "fanout_qa_studies",
    ],
)
def test_w1389_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
