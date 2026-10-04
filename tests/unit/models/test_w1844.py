import pytest


@pytest.mark.parametrize(
    "name",
    [
        "rimmon_qa_studies",
        "baalshamin_qa_studies",
        "ashima_qa_studies",
        "resheph2_qa_studies",
        "sahr_qa_studies",
        "aram2_qa_studies",
    ],
)
def test_w1844_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
