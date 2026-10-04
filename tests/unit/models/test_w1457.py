import pytest


@pytest.mark.parametrize(
    "name",
    [
        "beaver_qa_studies",
        "cougar_qa_studies",
        "bison_qa_studies",
        "elk_qa_studies",
        "lynx_qa_studies",
        "badger_qa_studies",
    ],
)
def test_w1457_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
