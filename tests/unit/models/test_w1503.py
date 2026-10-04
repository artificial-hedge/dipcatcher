import pytest


@pytest.mark.parametrize(
    "name",
    [
        "garter_qa_studies",
        "sidewinder_qa_studies",
        "racer_qa_studies",
        "keelback_qa_studies",
        "mockviper_qa_studies",
        "kingsnake_qa_studies",
    ],
)
def test_w1503_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
