import pytest


@pytest.mark.parametrize(
    "name",
    [
        "angra_qa_studies",
        "spenta_qa_studies",
        "haoma_qa_studies",
        "zal_qa_studies",
        "simurgh_qa_studies",
        "arash_qa_studies",
    ],
)
def test_w1769_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
