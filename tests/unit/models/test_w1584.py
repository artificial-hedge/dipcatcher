import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hare_qa_studies",
        "jackrabbit_qa_studies",
        "hyrax_qa_studies",
        "pika_qa_studies",
        "cottontail_qa_studies",
        "hedgehog_qa_studies",
    ],
)
def test_w1584_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
