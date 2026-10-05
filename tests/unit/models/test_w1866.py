import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ector_qa_studies",
        "hector_cameliard_qa_studies",
        "seneschal_qa_studies",
        "ynis_qa_studies",
        "avilion_qa_studies",
        "balan_qa_studies",
    ],
)
def test_w1866_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
