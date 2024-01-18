import unittest

from build_main_parser import build_main_parser
from commands.acdc import run_acdc
from utils.attr_dict import AttrDict
from utils.get_cases import get_cases


class RunACDCTest(unittest.TestCase):
  def test_acdc_runs_successfully_on_case_2(self):
    parser = build_main_parser()
    args = parser.parse_args(["run", "acdc", "-f", "-i=2", "--threshold=0.71"])
    case = get_cases(args)[0]
    run_acdc(case, args)
