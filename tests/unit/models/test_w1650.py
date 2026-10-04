import pytest


@pytest.mark.parametrize(
    "name",
    [
        "popobawa_qa_studies",
        "adjule_qa_studies",
        "agogwe_qa_studies",
        "rompo_qa_studies",
        "biloko_qa_studies",
        "kongamato_qa_studies",
    ],
)
def test_w1650_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
