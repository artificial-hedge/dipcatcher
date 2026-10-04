import pytest


@pytest.mark.parametrize(
    "name",
    [
        "enki_qa_studies",
        "enlil_qa_studies",
        "nanna_qa_studies",
        "utu_qa_studies",
        "tiamat_qa_studies",
        "abzu_qa_studies",
    ],
)
def test_w1691_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
