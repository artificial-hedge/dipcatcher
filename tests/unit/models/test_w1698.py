import pytest


@pytest.mark.parametrize(
    "name",
    [
        "maui_qa_studies",
        "pele_qa_studies",
        "tangaroa_qa_studies",
        "tane_qa_studies",
        "rangi_qa_studies",
        "menahune_qa_studies",
    ],
)
def test_w1698_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
