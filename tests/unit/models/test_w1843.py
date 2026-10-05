import pytest


@pytest.mark.parametrize(
    "name",
    [
        "dagon2_qa_studies",
        "ashtoreth_qa_studies",
        "baalzebub_qa_studies",
        "delilah_qa_studies",
        "goliath_qa_studies",
        "samson_qa_studies",
    ],
)
def test_w1843_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
