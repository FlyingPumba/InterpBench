import os
import unittest

import torch as t

from commands.analysis.compression_matrix import run
from commands.build_main_parser import build_main_parser


class CompressionMatrixAnalysisTest(unittest.TestCase):
  def test_compression_matrix_analysis_runs_successfully_on_case_3(self):
    # check that the needed files are in the right place, otherwise skip this test
    matrix_path = "../../case-00003-resid-8-compression-matrix.npy"
    if not os.path.exists(matrix_path):
      self.skipTest("case-00003-resid-8-compression-matrix.npy not found, skipping test")

    tracr_model_path = "../../benchmark/case-00003/tracr_model.pkl"
    if not os.path.exists(tracr_model_path):
      self.skipTest("case-00003-resid-8-compression-matrix.npy not found, skipping test")

    args, _ = build_main_parser().parse_known_args(["analysis", "compression-matrix",
                                                    "--matrix=" + matrix_path,
                                                    "--tracr-model=" + tracr_model_path,
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    run(args)