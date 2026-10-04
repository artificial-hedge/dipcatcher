import pytest


@pytest.mark.parametrize(
    "name",
    [
        "itzamna_qa_studies",
        "camazotz_qa_studies",
        "hunab_qa_studies",
        "ixmucane_qa_studies",
        "cabrakan_qa_studies",
        "zipacna_qa_studies",
    ],
)
def test_w1766_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
