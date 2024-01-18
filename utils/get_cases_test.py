import unittest

from utils.attr_dict import AttrDict
from utils.get_cases import get_cases_files


class GetCasesTest(unittest.TestCase):
  def test_all_cases(self):
    files = get_cases_files(None)
    self.assertEqual(len(files), 49)

  def test_cases_filtered_by_indices(self):
    args = AttrDict({"indices": "0,1,2"})
    files = get_cases_files(args)
    self.assertEqual(len(files), 3)