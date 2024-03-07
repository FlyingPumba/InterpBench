from iit.utils import index
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase

_tracr_ll_corrs = {
    '3': {
            ("is_x_3", None): {(0, "mlp", index.Ix[[None]])},
            ("frac_prevs_1", None): {(1, "attn", index.Ix[:, :, :2, :])},
    }
}

def get_tracr_ll_corr(case):
    if case.get_index() in _tracr_ll_corrs.keys():
        return _tracr_ll_corrs[case.get_index()]
    return None