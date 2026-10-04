import pytest


@pytest.mark.parametrize(
    "name",
    [
        "jorogumo_qa_studies",
        "nue_2_qa_studies",
        "kodama_qa_studies",
        "gashadokuro_qa_studies",
        "namahage_qa_studies",
        "tsuchinoko_qa_studies",
    ],
)
def test_w1636_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
