import pytest


@pytest.mark.parametrize(
    "name",
    [
        "pegasus_2_qa_studies",
        "nemean_qa_studies",
        "cerberus_2_qa_studies",
        "orthrus_qa_studies",
        "typhon_qa_studies",
        "argus_qa_studies",
    ],
)
def test_w1645_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
