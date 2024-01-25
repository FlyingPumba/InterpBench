import unittest

import torch as t

from commands.build_main_parser import build_main_parser
from commands.compile_benchmark import build_tracr_model, run_case_tests_on_tracr_model, build_transformer_lens_model, \
  run_case_tests_on_tl_model, compile_all
from utils.get_cases import get_cases


class CompileBenchmarkTest(unittest.TestCase):
  def test_all_cases_compile_successfully(self):
    args, _ = build_main_parser().parse_known_args(["compile", "-f"])
    cases = get_cases(args)
    for case in cases:
      print(f"\nCompiling {case}")
      tracr_output = build_tracr_model(case, force=args.force)
      # build_transformer_lens_model(case, args.force, tracr_output=tracr_output, device=args.device)

  def test_case_2_has_expected_outputs(self):
    args, _ = build_main_parser().parse_known_args(["compile",
                                                    "-f",
                                                    "-i=2",
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

  def test_case_3_has_expected_outputs(self):
    args, _ = build_main_parser().parse_known_args(["compile",
                                                    "-f",
                                                    "-i=3",
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

  def test_linear_compression_works_for_case_2(self):
    # Case 2 has a size of 117 for the residual stream. Let's try to compress it to 80.
    args, _ = build_main_parser().parse_known_args(["compile",
                                                    "-f",
                                                    "-i=2",
                                                    "--compress-residual=linear",
                                                    "--run-tests",
                                                    "--fail-on-error",
                                                    "--residual-stream-compression-size=80",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    compile_all(args)

  def test_linear_compression_works_for_case_3(self):
    # Case 3 has a size of 19 for the residual stream. Let's try to compress it to 14.
    args, _ = build_main_parser().parse_known_args(["compile",
                                                    "-f",
                                                    "-i=3",
                                                    "--compress-residual=linear",
                                                    "--run-tests",
                                                    "--fail-on-error",
                                                    "--residual-stream-compression-size=14",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    compile_all(args)
