import pytest


@pytest.mark.parametrize(
    "name",
    [
        "barnacle_qa_studies",
        "sandhopper_qa_studies",
        "copepod_qa_studies",
        "isopod_qa_studies",
        "amphipod_qa_studies",
        "krill_qa_studies",
    ],
)
def test_w1616_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
