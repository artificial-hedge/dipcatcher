import pytest


@pytest.mark.parametrize(
    "name",
    [
        "impala_qa_studies",
        "oryx_qa_studies",
        "kudu_qa_studies",
        "springbok_qa_studies",
        "eland_qa_studies",
        "antelope_qa_studies",
    ],
)
def test_w1478_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
