import pytest


@pytest.mark.parametrize(
    "name",
    [
        "itzamna_qa_studies",
        "kukulcan_qa_studies",
        "hunab_qa_studies",
        "chac_qa_studies",
        "ixchel_qa_studies",
        "yumkaax_qa_studies",
    ],
)
def test_w1702_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
