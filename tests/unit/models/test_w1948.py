import pytest


@pytest.mark.parametrize(
    "name",
    [
        "paimon_qa_studies",
        "asmodeus_qa_studies",
        "belial_qa_studies",
        "stolas_qa_studies",
        "astaroth_qa_studies",
        "furfur_qa_studies",
    ],
)
def test_w1948_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
