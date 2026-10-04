import pytest


@pytest.mark.parametrize(
    "name",
    [
        "arc_da_lite_studies",
        "emrqa_lite_studies",
        "drug_qa_lite_studies",
        "head_qa_lite_studies",
        "medmcqa_lite_studies",
        "ai2_arc_lite_studies",
    ],
)
def test_w1402_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
