from iit.utils import index
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase

_tracr_ll_corrs = {
    '3': {
            ("is_x_3", None): {(0, "mlp", index.Ix[[None]])},
            ("frac_prevs_1", None): {(1, "attn", index.Ix[:, :, 2, :])},
    }
}

def get_tracr_ll_corr(case):
    if isinstance(case, int):
        return _tracr_ll_corrs[str(case)] if str(case) in _tracr_ll_corrs.keys() else None
    if isinstance(case, str):
        return _tracr_ll_corrs[case] if case in _tracr_ll_corrs.keys() else None
    if isinstance(case, BenchmarkCase):
        if case.get_name() in _tracr_ll_corrs.keys():
            return _tracr_ll_corrs[case.get_name()]
        return None