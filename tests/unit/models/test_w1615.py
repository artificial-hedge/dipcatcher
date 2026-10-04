import pytest


@pytest.mark.parametrize(
    "name",
    [
        "slender_qa_studies",
        "slow_qa_studies",
        "golden_brown_qa_studies",
        "pygmy_qa_studies",
        "thin_spined_qa_studies",
        "gray_mouse_qa_studies",
    ],
)
def test_w1615_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
