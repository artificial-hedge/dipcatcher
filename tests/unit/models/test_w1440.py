import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cactus_qa_studies",
        "moss_qa_studies",
        "fern_qa_studies",
        "pine_qa_studies",
        "vine_qa_studies",
        "bamboo_qa_studies",
    ],
)
def test_w1440_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
