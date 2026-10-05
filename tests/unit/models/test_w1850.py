import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ayyur_qa_studies",
        "ifri_qa_studies",
        "tanit2_qa_studies",
        "anzar_qa_studies",
        "amma_qa_studies",
        "meghisen_qa_studies",
    ],
)
def test_w1850_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
