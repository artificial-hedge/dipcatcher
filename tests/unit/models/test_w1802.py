import pytest


@pytest.mark.parametrize(
    "name",
    [
        "kaikoura_qa_studies",
        "ranginui2_qa_studies",
        "tanemahuta2_qa_studies",
        "awhi2_qa_studies",
        "moana2_qa_studies",
        "hine3_qa_studies",
    ],
)
def test_w1802_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
