import pytest


@pytest.mark.parametrize(
    "name",
    [
        "nephthys_qa_studies",
        "serqet_qa_studies",
        "hapi_qa_studies",
        "tefnut_qa_studies",
        "khnum_qa_studies",
        "menhit_qa_studies",
    ],
)
def test_w1763_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
