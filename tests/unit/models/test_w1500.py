import pytest


@pytest.mark.parametrize(
    "name",
    [
        "centipede_qa_studies",
        "horntail_qa_studies",
        "millipede_qa_studies",
        "lacewing_qa_studies",
        "caddisfly_qa_studies",
        "spider_qa_studies",
    ],
)
def test_w1500_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
