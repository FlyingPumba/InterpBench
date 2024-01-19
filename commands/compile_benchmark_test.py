import unittest

from commands.compile_benchmark import build_tracr_model, run_case_tests_on_tracr_model, build_transformer_lens_model, \
  run_case_tests_on_tl_model
from utils.attr_dict import AttrDict
from utils.get_cases import get_cases


class CompileBenchmarkTest(unittest.TestCase):
  def test_all_cases_compile_successfully(self):
    cases = get_cases(None)
    for case in cases:
      print(f"\nCompiling {case}")
      tracr_output = build_tracr_model(case, force=True)
      # build_transformer_lens_model(case, args.force, tracr_output=tracr_output)

  def test_case_2_has_expected_outputs(self):
    args = AttrDict({"indices": "2", "force": True})
    cases = get_cases(args)
    for case in cases:
      print(f"\nCompiling {case}")
      tracr_output = build_tracr_model(case, args.force)
      run_case_tests_on_tracr_model(case, tracr_output.model)
      tl_model = build_transformer_lens_model(case, args.force, tracr_output=tracr_output)
      run_case_tests_on_tl_model(case, tl_model)
