import unittest
from pathlib import Path

import torch as t

from circuits_benchmark.benchmark.cases.case_3 import Case3
from circuits_benchmark.commands.build_main_parser import build_main_parser
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.utils.project_paths import detect_project_root


class CompileBenchmarkTest(unittest.TestCase):
  def test_all_cases_can_be_compiled_and_have_expected_outputs(self):
    project_root = detect_project_root()
    case_file_names = [str(f.name) for f in Path(project_root).glob("benchmark/cases/case_*.py") if f.is_file()]
    indices = [f.split("_")[1].split(".")[0] for f in case_file_names]

    # remove cases that are known to fail and we have yet to fix
    failing_cases = ["8", "10", "11", "16", "18", "20", "23", "36"]
    for failing_case in failing_cases:
      if failing_case in indices:
        indices.remove(failing_case)

    args, _ = build_main_parser().parse_known_args(["compile",
                                                    ("-i=" + ",".join(indices)),
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
