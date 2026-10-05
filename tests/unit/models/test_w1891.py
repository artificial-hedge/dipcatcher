import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bergelmir_qa_studies",
        "jormunrek_qa_studies",
        "khan_tengri_qa_studies",
        "peri_qa_studies",
        "umm_sibyan_qa_studies",
        "ymir_hrimthurs_qa_studies",
    ],
)
def test_w1891_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
