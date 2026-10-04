import pytest


@pytest.mark.parametrize(
    "name",
    [
        "afanc_qa_studies",
        "eachuisge_qa_studies",
        "aatxe_qa_studies",
        "achiyalabopa_qa_studies",
        "akhlut_qa_studies",
        "amarok_qa_studies",
    ],
)
def test_w1656_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
