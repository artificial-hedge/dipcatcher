import pytest


@pytest.mark.parametrize(
    "name",
    [
        "tropicbird_qa_studies",
        "fulmar_qa_studies",
        "auk_qa_studies",
        "kittiwake_qa_studies",
        "jaeger_qa_studies",
        "gull_qa_studies",
    ],
)
def test_w1487_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
