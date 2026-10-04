import pytest


@pytest.mark.parametrize(
    "name",
    [
        "janus_qa_studies",
        "solinvictus_qa_studies",
        "flora_qa_studies",
        "pomona_qa_studies",
        "silvanus_qa_studies",
        "ceres_qa_studies",
    ],
)
def test_w1788_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
