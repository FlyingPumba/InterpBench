import os
import unittest

from circuits_benchmark.benchmark.cases.case_3 import Case3
from circuits_benchmark.benchmark.cases.case_ioi import CaseIOI
from circuits_benchmark.commands.algorithms.acdc import ACDCRunner, ACDCConfig
from circuits_benchmark.commands.build_main_parser import build_main_parser
from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import get_ll_model_loader
from circuits_benchmark.utils.project_paths import get_default_output_dir
from circuits_benchmark.commands.train import train

class ACDCTest(unittest.TestCase):
  def setup_method(self, test_method):
    # detect if SIIT and Natural model for Case 3 are available, and train them if not
    output_dir = get_default_output_dir()

    natural_model_path = f"{output_dir}/ll_models/3/ll_model_100.pth"
    if not os.path.exists(natural_model_path):
      # train natural model
      args, _ = build_main_parser().parse_known_args(["train",
                                                      "iit",
                                                      "-i=3",
                                                      "--epochs=1",
                                                      "--early-stop",
                                                      "-s=0",
                                                      "-iit=0",
                                                      "--num-samples=10",
                                                      "--device=cpu"])
      train.run(args)
    assert os.path.exists(natural_model_path)

    siit_model_path = f"{output_dir}/ll_models/3/ll_model_510.pth"
    if not os.path.exists(siit_model_path):
      # train SIIT model
      args, _ = build_main_parser().parse_known_args(["train",
                                                      "iit",
                                                      "-i=3",
                                                      "--epochs=1",
                                                      "--early-stop",
                                                      "--num-samples=10",
                                                      "--device=cpu"])
      train.run(args)
    assert os.path.exists(siit_model_path)

  def test_acdc_works_on_siit_model_for_case_3(self):
    case = Case3()
    config = ACDCConfig(
      threshold=0.001,
      data_size=10,
    )
    ll_model_loader = get_ll_model_loader(
      case,
      natural=False,
      tracr=False,
      interp_bench=False,
      siit_weights="510",
      load_from_wandb=False
    )
    circuit, circuit_eval_result = ACDCRunner(case, config=config).run_using_model_loader(ll_model_loader)
    assert circuit is not None
    assert circuit_eval_result is not None

  def test_acdc_works_on_natural_model_for_case_3(self):
    case = Case3()
    config = ACDCConfig(
      threshold=0.001,
      data_size=10,
    )
    ll_model_loader = get_ll_model_loader(
      case,
      natural=True,
      tracr=False,
      interp_bench=False,
      siit_weights=None,
      load_from_wandb=False
    )
    circuit, circuit_eval_result = ACDCRunner(case, config=config).run_using_model_loader(ll_model_loader)
    assert circuit is not None
    assert circuit_eval_result is not None

  def test_acdc_works_on_tracr_model_for_case_3(self):
    case = Case3()
    config = ACDCConfig(
      threshold=0.001,
      data_size=10,
    )
    ll_model_loader = get_ll_model_loader(
      case,
      natural=False,
      tracr=True,
      interp_bench=False,
      siit_weights=None,
      load_from_wandb=False
    )
    circuit, circuit_eval_result = ACDCRunner(case, config=config).run_using_model_loader(ll_model_loader)
    assert circuit is not None
    assert circuit_eval_result is not None

  def test_acdc_works_on_interp_bench_model_for_case_3(self):
    case = Case3()
    config = ACDCConfig(
      threshold=0.001,
      data_size=10,
    )
    ll_model_loader = get_ll_model_loader(
      case,
      natural=False,
      tracr=False,
      interp_bench=True,
      siit_weights=None,
      load_from_wandb=False
    )
    circuit, circuit_eval_result = ACDCRunner(case, config=config).run_using_model_loader(ll_model_loader)
    assert circuit is not None
    assert circuit_eval_result is not None

  def test_acdc_works_on_interp_bench_model_for_case_ioi(self):
    case = CaseIOI()
    config = ACDCConfig(
      threshold=0.001,
      data_size=10,
    )
    ll_model_loader = get_ll_model_loader(
      case,
      natural=False,
      tracr=False,
      interp_bench=True,
      siit_weights=None,
      load_from_wandb=False
    )
    circuit, circuit_eval_result = ACDCRunner(case, config=config).run_using_model_loader(ll_model_loader)
    assert circuit is not None
    assert circuit_eval_result is not None
