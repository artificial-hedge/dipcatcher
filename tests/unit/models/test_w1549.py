import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cassowary_qa_studies",
        "kiwi_qa_studies",
        "rhea_qa_studies",
        "tinamou_qa_studies",
        "emu_qa_studies",
        "ostrich_qa_studies",
    ],
)
def test_w1549_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
