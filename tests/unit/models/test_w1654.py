import pytest


@pytest.mark.parametrize(
    "name",
    [
        "leucrotta_qa_studies",
        "bonnacon_qa_studies",
        "amphisbaena_qa_studies",
        "parandrus_qa_studies",
        "questing_qa_studies",
        "cerastes_qa_studies",
    ],
)
def test_w1654_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
