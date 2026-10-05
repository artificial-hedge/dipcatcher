import pytest


@pytest.mark.parametrize(
    "name",
    [
        "el3_qa_studies",
        "baal3_qa_studies",
        "asherah3_qa_studies",
        "mot3_qa_studies",
        "lotan3_qa_studies",
        "kothar3_qa_studies",
    ],
)
def test_w1831_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
