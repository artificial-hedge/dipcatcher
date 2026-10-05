import pytest


@pytest.mark.parametrize(
    "name",
    [
        "orobas_qa_studies",
        "gremory_qa_studies",
        "ose_qa_studies",
        "amy_qa_studies",
        "murmur_qa_studies",
        "andrealphus_qa_studies",
    ],
)
def test_w1953_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
