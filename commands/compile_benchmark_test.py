import unittest

from commands.compile_benchmark import build_tracr_model, run_case_tests_on_tracr_model
from utils.attr_dict import AttrDict
from utils.get_cases import get_cases


class CompileBenchmarkTest(unittest.TestCase):
  def test_all_cases_compile_successfully(self):
    cases = get_cases(None)
    for case in cases:
      print(f"\nCompiling {case}")
      tracr_output = build_tracr_model(case, force=True)
      # build_transformer_lens_model(file_path, args.force, tracr_output=tracr_output)

  def test_all_cases_have_expected_outputs(self):
    cases = get_cases(None)
    for case in cases:
      print(f"\nCompiling {case}")
      tracr_output = build_tracr_model(case)
      run_case_tests_on_tracr_model(case, tracr_output.model)