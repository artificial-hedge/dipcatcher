import pytest


@pytest.mark.parametrize(
    "name",
    [
        "yamenna_qa_studies",
        "weded_qa_studies",
        "bozrum_qa_studies",
        "guillyn_qa_studies",
        "baal_marod_qa_studies",
        "hammonites_qa_studies",
    ],
)
def test_w1878_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
