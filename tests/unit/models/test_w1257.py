"""Wave 1257 clinical-lab canon — unit tests."""

from quant_fund.models import (
    culture_studies,
    flow_cytometry_studies,
    immunoassay_studies,
    microscopy_studies,
    pcr_studies,
    serology_studies,
)

_FAMILIES = (
    (immunoassay_studies, "immunoassay_studies"),
    (pcr_studies, "pcr_studies"),
    (serology_studies, "serology_studies"),
    (culture_studies, "culture_studies"),
    (microscopy_studies, "microscopy_studies"),
    (flow_cytometry_studies, "flow_cytometry_studies"),
)


def test_w1257_benches_return_unit_floats() -> None:
    for mod, name in _FAMILIES:
        out = getattr(mod, f"bench_{name}")(seed=0)
        assert len(out) == 1
        ((k, v),) = out.items()
        assert k == f"synthetic_{name}"
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0
