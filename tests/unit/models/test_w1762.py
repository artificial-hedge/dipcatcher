import pytest


@pytest.mark.parametrize(
    "name",
    [
        "baldr_qa_studies",
        "heimdall_qa_studies",
        "tyr_qa_studies",
        "freya_qa_studies",
        "idunn_qa_studies",
        "bragi_qa_studies",
    ],
)
def test_w1762_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
