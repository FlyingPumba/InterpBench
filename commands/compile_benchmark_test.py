import unittest

import torch as t

from commands.build_main_parser import build_main_parser
from commands.compile_benchmark import build_tracr_model, run_case_tests_on_tracr_model, build_transformer_lens_model, \
  run_case_tests_on_tl_model, compile_all
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
                                                    "-f",
                                                    ("-i=" + ",".join(indices)),
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    cases = get_cases(args)
    for case in cases:
      print(f"\nCompiling {case}")
      tracr_output = build_tracr_model(case, args.force)
      run_case_tests_on_tracr_model(case, tracr_output.model)
      tl_model = build_transformer_lens_model(case,
                                              force=args.force,
                                              tracr_output=tracr_output,
                                              device=args.device)
      run_case_tests_on_tl_model(case, tl_model)

  def test_linear_compression_does_not_throw_exceptions_on_any_case(self):
    args, _ = build_main_parser().parse_known_args(["compile",
                                                    "-f",
                                                    "-i=2,3",
                                                    "--compress-residual=linear",
                                                    "--fail-on-error",
                                                    "--residual-stream-compression-size=5",
                                                    "--epochs=2",
                                                    "--train-data-size=10",
                                                    "--test-data-ratio=0.3",
                                                    "--batch-size=2",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    compile_all(args)

  def test_auto_linear_compression_works_for_case_2(self):
    # Case 2 has a size of 117 for the residual stream. Let's try to compress it to 80.
    args, _ = build_main_parser().parse_known_args(["compile",
                                                    "-i=2",
                                                    "--compress-residual=linear",
                                                    "--residual-stream-compression-size=auto",
                                                    "-wandb-project=compression",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    compile_all(args)
