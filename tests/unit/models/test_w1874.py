import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ushtey_qa_studies",
        "arkan_sonney_qa_studies",
        "dozmary_qa_studies",
        "sleih_beggey_qa_studies",
        "shooil_ghoul_qa_studies",
        "loaghtan_qa_studies",
    ],
)
def test_w1874_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
