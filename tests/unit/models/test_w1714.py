import pytest


@pytest.mark.parametrize(
    "name",
    [
        "arimasp_qa_studies",
        "tavrita_qa_studies",
        "argimpasa_qa_studies",
        "papaios_qa_studies",
        "tabiti_qa_studies",
        "thagimasadas_qa_studies",
    ],
)
def test_w1714_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
