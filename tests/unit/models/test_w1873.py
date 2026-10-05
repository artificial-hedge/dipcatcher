import pytest


@pytest.mark.parametrize(
    "name",
    [
        "paotr_bugel_qa_studies",
        "noz_vat_qa_studies",
        "santez_nonna_qa_studies",
        "kaier_qa_studies",
        "darkman_qa_studies",
        "ar_marzh_qa_studies",
    ],
)
def test_w1873_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
