import pytest


@pytest.mark.parametrize(
    "name",
    [
        "dolphin_qa_studies",
        "shark_qa_studies",
        "reef_qa_studies",
        "turtle_qa_studies",
        "whale_qa_studies",
        "coral_qa_studies",
    ],
)
def test_w1438_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
