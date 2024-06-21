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
    case_file_names = [str(f.name) for f in Path(project_root).glob("circuits_benchmark/benchmark/cases/case_*.py") if f.is_file()]
    indices = [f.split("_")[1].split(".")[0] for f in case_file_names]

    args, _ = build_main_parser().parse_known_args(["compile",
                                                    ("-i=" + ",".join(indices)),
                                                    "--fail-on-error",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    cases = get_cases(args)
    for case in cases:
      if case.get_name() in ["16", "27", "38"]:
        continue

      print(f"\nCompiling {case}")
      tracr_output = case.build_tracr_model()
      case.run_case_tests_on_tracr_model(tracr_model=tracr_output.model)
      tl_model = case.build_transformer_lens_model(tracr_model=tracr_output.model, device=args.device)
      case.run_case_tests_on_tl_model(tl_model=tl_model)

      # print some stats about the model
      print(f"Is categorical status: {tl_model.is_categorical()}")
      print(f"Number of layers: {len(tl_model.blocks)}")
      max_heads = max([len(b.attn.W_Q) for b in tl_model.blocks])
      print(f"Max number of attention heads: {max_heads}")

  def test_cases_can_be_instantiated_directly(self):
    case = Case3()
    assert case.get_name() == "3"
