import unittest

import torch as t

from commands.compile_benchmark import build_tracr_model, run_case_tests_on_tracr_model, build_transformer_lens_model, \
  run_case_tests_on_tl_model, compile_all
from utils.attr_dict import AttrDict
from utils.get_cases import get_cases


class CompileBenchmarkTest(unittest.TestCase):
  def test_all_cases_compile_successfully(self):
    cases = get_cases(None)
    for case in cases:
      print(f"\nCompiling {case}")
      tracr_output = build_tracr_model(case, force=True)
      # build_transformer_lens_model(case, args.force, tracr_output=tracr_output, device=args.device)

  def test_case_2_has_expected_outputs(self):
    args = AttrDict({"indices": "2", "force": True,
                     "device": "cuda" if t.cuda.is_available() else "cpu"})
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
    args = AttrDict({"indices": "2",
                     "force": True,
                     "compress_residual": "linear",
                     "run_tests": True,
                     "fail_on_error": True,
                     "residual_stream_compression_size": 80,
                     "device": "cuda" if t.cuda.is_available() else "cpu"})
    compile_all(args)