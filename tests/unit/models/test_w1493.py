import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hartebeest_qa_studies",
        "waterbuck_qa_studies",
        "topi_qa_studies",
        "duiker_qa_studies",
        "bongo_qa_studies",
        "nyala_qa_studies",
    ],
)
def test_w1493_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
