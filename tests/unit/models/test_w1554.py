import pytest


@pytest.mark.parametrize(
    "name",
    [
        "wolf_spider_qa_studies",
        "orb_weaver_qa_studies",
        "huntsman_qa_studies",
        "jumping_spider_qa_studies",
        "black_widow_qa_studies",
        "tarantula_qa_studies",
    ],
)
def test_w1554_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
