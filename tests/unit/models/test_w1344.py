import pytest


@pytest.mark.parametrize(
    "name",
    [
        "race_qa_studies",
        "dream_qa_studies",
        "mctest_qa_studies",
        "qasper_qa_studies",
        "duorc_qa_studies",
        "boolq_qa_studies",
    ],
)
def test_w1344_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
