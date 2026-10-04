import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ifrit_qa_studies",
        "ghul_qa_studies",
        "marid_qa_studies",
        "shaitan_qa_studies",
        "jann_qa_studies",
        "nasnas_qa_studies",
    ],
)
def test_w1914_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
