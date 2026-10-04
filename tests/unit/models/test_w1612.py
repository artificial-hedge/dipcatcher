import pytest


@pytest.mark.parametrize(
    "name",
    [
        "argali_qa_studies",
        "bighorn_qa_studies",
        "llama_qa_studies",
        "urial_qa_studies",
        "dall_qa_studies",
        "mouflon_qa_studies",
    ],
)
def test_w1612_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
