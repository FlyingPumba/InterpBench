import unittest

import torch as t

from benchmark.cases.case_3 import Case3
from commands.build_main_parser import build_main_parser
from utils.get_cases import get_cases


class CompileBenchmarkTest(unittest.TestCase):
  def test_all_cases_can_be_compiled_and_have_expected_outputs(self):
    indices = [str(i) for i in range(1, 48)]

    # remove cases that are known to fail and we have yet to fix
    failing_cases = ["1", "6", "8", "10", "11", "12", "16", "17", "18", "19", "20", "23", "25", "30", "31", "32", "33",
                     "34", "36", "38", "39", "41", "44", "46", "47"]
    for failing_case in failing_cases:
      if failing_case in indices:
        indices.remove(failing_case)

    args, _ = build_main_parser().parse_known_args(["compile",
                                                    ("-i=" + ",".join(indices)),
                                                    "-f",
                                                    "--fail-on-error",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    cases = get_cases(args)
    for case in cases:
      print(f"\nCompiling {case}")
      tracr_output = case.build_tracr_model()
      case.run_case_tests_on_tracr_model(tracr_model=tracr_output.model)
      tl_model = case.build_transformer_lens_model(tracr_model=tracr_output.model, device=args.device)
      case.run_case_tests_on_tl_model(tl_model=tl_model)

  def test_cases_can_be_instantiated_directly(self):
    case = Case3()
    assert case.get_index() == "3"
