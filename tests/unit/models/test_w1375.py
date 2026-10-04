import pytest


@pytest.mark.parametrize(
    "name",
    [
        "facet_sum_studies",
        "sci_lay_studies",
        "patent_sum_studies",
        "scitldr_lite_studies",
        "spectrum_sum_studies",
        "ms2_lite_studies",
    ],
)
def test_w1375_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
