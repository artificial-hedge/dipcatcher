import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bigfoot_qa_studies",
        "loch_ness_qa_studies",
        "bunyip_qa_studies",
        "wendigo_qa_studies",
        "skinwalker_qa_studies",
        "rougarou_qa_studies",
    ],
)
def test_w1630_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
