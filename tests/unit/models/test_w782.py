from quant_fund.models.ffusion_lims import (
    bench_ffusion_lims,
)
from quant_fund.models.filt_proc import bench_filt_proc
from quant_fund.models.jacod_shiryaev import (
    bench_jacod_shiryaev,
)
from quant_fund.models.kunita_watanabe import (
    bench_kunita_watanabe,
)
from quant_fund.models.pinsky_proc import bench_pinsky_proc
from quant_fund.models.slivnyak import bench_slivnyak


def test_pinsky_proc():
    assert bench_pinsky_proc()["synthetic_pinsky_proc"] == 1.0


def test_ffusion_lims():
    assert bench_ffusion_lims()["synthetic_ffusion_lims"] == 1.0


def test_kunita_watanabe():
    assert bench_kunita_watanabe()["synthetic_kunita_watanabe"] == 1.0


def test_filt_proc():
    assert bench_filt_proc()["synthetic_filt_proc"] == 1.0


def test_slivnyak():
    assert bench_slivnyak()["synthetic_slivnyak"] == 1.0


def test_jacod_shiryaev():
    assert bench_jacod_shiryaev()["synthetic_jacod_shiryaev"] == 1.0
