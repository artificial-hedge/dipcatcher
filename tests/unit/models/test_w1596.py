import pytest


@pytest.mark.parametrize(
    "name",
    [
        "false_killer_qa_studies",
        "pygmy_whale_qa_studies",
        "sea_lion_qa_studies",
        "bottlenose_qa_studies",
        "dusky_dolphin_qa_studies",
        "melon_head_qa_studies",
    ],
)
def test_w1596_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
