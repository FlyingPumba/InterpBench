import unittest

from utils.attr_dict import AttrDict
from utils.get_cases import get_cases


class GetCasesTest(unittest.TestCase):
  def test_cases_filtered_by_indices(self):
    args = AttrDict({"indices": "1,2,3"})
    cases = get_cases(args)
    self.assertEqual(len(cases), 3)