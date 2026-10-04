import pytest


@pytest.mark.parametrize(
    "name",
    [
        "sapsucker_qa_studies",
        "pileated_qa_studies",
        "flicker_qa_studies",
        "downy_qa_studies",
        "wryneck_qa_studies",
        "woodpecker_qa_studies",
    ],
)
def test_w1531_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
