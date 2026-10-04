import pytest


@pytest.mark.parametrize(
    "name",
    [
        "oya_qa_studies",
        "shango_qa_studies",
        "obatala_qa_studies",
        "ifa_qa_studies",
        "anansi_qa_studies",
        "abiku_qa_studies",
    ],
)
def test_w1758_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
