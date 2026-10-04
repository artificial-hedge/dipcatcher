import pytest


@pytest.mark.parametrize(
    "name",
    [
        "gana_qa_studies",
        "danava_qa_studies",
        "kalakeya_qa_studies",
        "kimpurusha_qa_studies",
        "gandharva_qa_studies",
        "rakshasa_qa_studies",
    ],
)
def test_w1683_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
