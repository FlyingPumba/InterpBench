import os
import unittest

import torch as t

from benchmark.cases.case_3 import Case3
from commands.analysis.compression_matrix import run
from commands.build_main_parser import build_main_parser
from utils.detect_project_root import detect_project_root


class CompressionMatrixAnalysisTest(unittest.TestCase):
  def test_compression_matrix_analysis_runs_successfully_on_case_3(self):
    project_root = detect_project_root()

    # check that the needed files are in the right place, otherwise skip this test
    matrix_path = os.path.join(project_root, "results", "case-3-resid-5-compression-matrix.npy")
    if not os.path.exists(matrix_path):
      self.skipTest("case-3-resid-5-compression-matrix.npy not found, skipping test")

    case3 = Case3()
    tracr_model_path = case3.get_tracr_model_pickle_path()
    if not os.path.exists(tracr_model_path):
      self.skipTest("case 3's tracr model not found, skipping test")

    args, _ = build_main_parser().parse_known_args(["analysis", "compression-matrix",
                                                    "--matrix=" + matrix_path,
                                                    "--tracr-model=" + tracr_model_path,
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    run(args)