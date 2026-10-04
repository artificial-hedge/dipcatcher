import pytest


@pytest.mark.parametrize(
    "name",
    [
        "commonsense_lite_studies",
        "muin_lite_studies",
        "mr_lite_studies",
        "qasc_sci2_studies",
        "winogrande_lite_studies",
        "logi_qa_studies",
    ],
)
def test_w1373_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
