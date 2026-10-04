import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ahuramazda_qa_studies",
        "ahriman_qa_studies",
        "mithra_qa_studies",
        "anahita_qa_studies",
        "verethragna_qa_studies",
        "yazata_qa_studies",
    ],
)
def test_w1705_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
