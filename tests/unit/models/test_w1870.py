import pytest


@pytest.mark.parametrize(
    "name",
    [
        "spriggan_qa_studies",
        "bucca_qa_studies",
        "knocker_qa_studies",
        "piskie_qa_studies",
        "tregeagle_qa_studies",
        "morgawr_qa_studies",
    ],
)
def test_w1870_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
