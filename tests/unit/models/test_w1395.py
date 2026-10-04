import pytest


@pytest.mark.parametrize(
    "name",
    [
        "fine_qa_studies",
        "kwik_qa_studies",
        "hotpot2_studies",
        "quest_qa_studies",
        "tatqa2_studies",
        "bamboogle_studies",
    ],
)
def test_w1395_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
