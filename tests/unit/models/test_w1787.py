import pytest


@pytest.mark.parametrize(
    "name",
    [
        "odin_qa_studies",
        "thor_qa_studies",
        "heimdal_qa_studies",
        "norns_qa_studies",
        "valkyrie_qa_studies",
        "eir_qa_studies",
    ],
)
def test_w1787_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
