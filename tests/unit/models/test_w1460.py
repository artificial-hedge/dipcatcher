import pytest


@pytest.mark.parametrize(
    "name",
    [
        "elephant_qa_studies",
        "giraffe_qa_studies",
        "gazelle_qa_studies",
        "wildebeest_qa_studies",
        "zebra_qa_studies",
        "baboon_qa_studies",
    ],
)
def test_w1460_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
