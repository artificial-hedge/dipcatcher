import pytest


@pytest.mark.parametrize(
    "name",
    [
        "pombero_qa_studies",
        "kurupi_qa_studies",
        "mboitui_qa_studies",
        "aoao_qa_studies",
        "jasy_jatere_qa_studies",
        "luison_qa_studies",
    ],
)
def test_w1936_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
