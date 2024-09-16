import unittest

from circuits_benchmark.benchmark.cases.case_3 import Case3
from circuits_benchmark.benchmark.cases.case_ioi import CaseIOI
from circuits_benchmark.commands.algorithms.sp import SPRunner, SPConfig
from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import get_ll_model_loader
from tests.utils import setup_iit_models


class SPTest(unittest.TestCase):
    def setup_method(self, test_method):
        setup_iit_models()

    def test_sp_works_on_siit_model_for_case_3(self):
        case = Case3()
        config = SPConfig(
            lambda_reg=0.001,
            data_size=10,
            epochs=1
        )
        ll_model_loader = get_ll_model_loader(
            case,
            natural=False,
            tracr=False,
            interp_bench=False,
            siit_weights="510",
            load_from_wandb=False
        )
        circuit, circuit_eval_result = SPRunner(case, config=config).run_using_model_loader(ll_model_loader)
        assert circuit is not None
        assert circuit_eval_result is not None

    def test_sp_works_on_natural_model_for_case_3(self):
        case = Case3()
        config = SPConfig(
            lambda_reg=0.001,
            data_size=10,
            epochs=1
        )
        ll_model_loader = get_ll_model_loader(
            case,
            natural=True,
            tracr=False,
            interp_bench=False,
            siit_weights=None,
            load_from_wandb=False
        )
        circuit, circuit_eval_result = SPRunner(case, config=config).run_using_model_loader(ll_model_loader)
        assert circuit is not None
        assert circuit_eval_result is not None

    def test_sp_works_on_tracr_model_for_case_3(self):
        case = Case3()
        config = SPConfig(
            lambda_reg=0.001,
            data_size=10,
            epochs=1
        )
        ll_model_loader = get_ll_model_loader(
            case,
            natural=False,
            tracr=True,
            interp_bench=False,
            siit_weights=None,
            load_from_wandb=False
        )
        circuit, circuit_eval_result = SPRunner(case, config=config).run_using_model_loader(ll_model_loader)
        assert circuit is not None
        assert circuit_eval_result is not None

    def test_sp_works_on_interp_bench_model_for_case_3(self):
        case = Case3()
        config = SPConfig(
            lambda_reg=0.001,
            data_size=10,
            epochs=1
        )
        ll_model_loader = get_ll_model_loader(
            case,
            natural=False,
            tracr=False,
            interp_bench=True,
            siit_weights=None,
            load_from_wandb=False
        )
        circuit, circuit_eval_result = SPRunner(case, config=config).run_using_model_loader(ll_model_loader)
        assert circuit is not None
        assert circuit_eval_result is not None

    def test_sp_works_on_interp_bench_model_for_case_ioi(self):
        case = CaseIOI()
        config = SPConfig(
            lambda_reg=0.001,
            data_size=10,
            epochs=1
        )
        ll_model_loader = get_ll_model_loader(
            case,
            natural=False,
            tracr=False,
            interp_bench=True,
            siit_weights=None,
            load_from_wandb=False
        )
        circuit, circuit_eval_result = SPRunner(case, config=config).run_using_model_loader(ll_model_loader)
        assert circuit is not None
        assert circuit_eval_result is not None
