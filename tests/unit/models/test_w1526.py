import pytest


@pytest.mark.parametrize(
    "name",
    [
        "horsetail_qa_studies",
        "swordfern_qa_studies",
        "maidenhair_qa_studies",
        "staghorn_qa_studies",
        "treefern_qa_studies",
        "bracken_qa_studies",
    ],
)
def test_w1526_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
