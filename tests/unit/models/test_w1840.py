import pytest


@pytest.mark.parametrize(
    "name",
    [
        "qos2_qa_studies",
        "chemosh2_qa_studies",
        "haddad2_qa_studies",
        "mot2_qa_studies",
        "atargatis2_qa_studies",
        "gad2_qa_studies",
    ],
)
def test_w1840_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
