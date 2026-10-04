import pytest


@pytest.mark.parametrize(
    "name",
    [
        "vili_qa_studies",
        "ve_qa_studies",
        "honir_qa_studies",
        "lodurr_qa_studies",
        "mimir_qa_studies",
        "kvasir_qa_studies",
    ],
)
def test_w1723_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
