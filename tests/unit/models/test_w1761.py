import pytest


@pytest.mark.parametrize(
    "name",
    [
        "enlil_qa_studies",
        "enki_qa_studies",
        "inanna_qa_studies",
        "nanna_qa_studies",
        "utu_qa_studies",
        "ninhursag_qa_studies",
    ],
)
def test_w1761_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
