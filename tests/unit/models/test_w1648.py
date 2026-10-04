import pytest


@pytest.mark.parametrize(
    "name",
    [
        "zilant_qa_studies",
        "aitvaras_qa_studies",
        "indus_qa_studies",
        "bilwis_qa_studies",
        "viy_qa_studies",
        "kudlak_qa_studies",
    ],
)
def test_w1648_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
