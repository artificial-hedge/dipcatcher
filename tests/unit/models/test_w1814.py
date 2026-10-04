import pytest


@pytest.mark.parametrize(
    "name",
    [
        "tiamat2_qa_studies",
        "marduk2_qa_studies",
        "enki2_qa_studies",
        "ninhursag2_qa_studies",
        "utu2_qa_studies",
        "nanna2_qa_studies",
    ],
)
def test_w1814_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
