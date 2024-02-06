import unittest

import torch as t

from commands.algorithms.acdc import run_acdc
from commands.build_main_parser import build_main_parser
from utils.get_cases import get_cases


class RunACDCTest(unittest.TestCase):
  def test_acdc_runs_successfully_on_case_2(self):
    args, _ = build_main_parser().parse_known_args(["run", "acdc",
                                                    "-i=2",
                                                    "--metric=l2",
                                                    "--threshold=0.028",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    case = get_cases(args)[0]
    run_acdc(case, args)

  def test_acdc_fails_if_using_kl_on_case_2(self):
    args, _ = build_main_parser().parse_known_args(["run", "acdc",
                                                    "-i=2",
                                                    "--metric=kl",
                                                    "--threshold=0.028",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    case = get_cases(args)[0]

    with self.assertRaises(ValueError):
      run_acdc(case, args)

  def test_acdc_runs_successfully_on_case_3_using_l2(self):
    args, _ = build_main_parser().parse_known_args(["run", "acdc",
                                                    "-i=3",
                                                    "--metric=l2",
                                                    "--threshold=0.028",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    case = get_cases(args)[0]
    run_acdc(case, args)
