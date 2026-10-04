import pytest


@pytest.mark.parametrize(
    "name",
    [
        "blind_salamander_qa_studies",
        "cave_spider_qa_studies",
        "cave_shrimp_qa_studies",
        "grotto_salamander_qa_studies",
        "proteus_qa_studies",
        "cave_swiftlet_qa_studies",
    ],
)
def test_w1625_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
