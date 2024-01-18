import unittest

from commands.compile_benchmark import build_tracr_model
from utils.get_cases import get_cases


class CompileBenchmarkTest(unittest.TestCase):
  def test_all_cases_compile_successfully(self):
    cases = get_cases(None)
    for case in cases:
      print(f"\nCompiling {case}")
      tracr_output = build_tracr_model(case, force=True)
      # build_transformer_lens_model(file_path, args.force, tracr_output=tracr_output)
