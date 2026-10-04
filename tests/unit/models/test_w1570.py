import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ghost_crab_qa_studies",
        "spider_crab_qa_studies",
        "porcelain_qa_studies",
        "fiddler_crab_qa_studies",
        "horseshoe_qa_studies",
        "mud_crab_qa_studies",
    ],
)
def test_w1570_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
