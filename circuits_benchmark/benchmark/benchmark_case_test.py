import unittest

from circuits_benchmark.benchmark.cases.case_3 import Case3


class BenchmarkCaseTest(unittest.TestCase):
  def test_get_all_clean_data(self):
    case = Case3()
    data = case.get_clean_data(count=None)
    assert len(data.get_inputs()) == 320

  def test_get_partial_clean_data(self):
    case = Case3()
    data = case.get_clean_data(count=10)
    assert len(data.get_inputs()) == 10
