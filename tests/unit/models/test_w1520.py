import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bolete_qa_studies",
        "morel_qa_studies",
        "chanterelle_qa_studies",
        "puffball_qa_studies",
        "inkcap_qa_studies",
        "agaric_qa_studies",
    ],
)
def test_w1520_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
