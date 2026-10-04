import pytest


@pytest.mark.parametrize(
    "name",
    [
        "yowie_qa_studies",
        "minka_qa_studies",
        "yara_qa_studies",
        "papin_qa_studies",
        "awgy_qa_studies",
        "kuritja_qa_studies",
    ],
)
def test_w1651_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
