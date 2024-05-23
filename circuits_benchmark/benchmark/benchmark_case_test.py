import unittest
from typing import List

from circuits_benchmark.benchmark.cases.case_1 import Case1
from circuits_benchmark.benchmark.cases.case_3 import Case3
from circuits_benchmark.benchmark.cases.case_5 import Case5


class BenchmarkCaseTest(unittest.TestCase):
  def test_get_all_clean_data(self):
    case = Case3()
    data = case.get_clean_data(count=None, variable_length_seqs=True)
    assert len(data.get_inputs()) == 320

  def test_get_partial_clean_data(self):
    case = Case3()
    data = case.get_clean_data(count=10, variable_length_seqs=True)
    assert len(data.get_inputs()) == 10

  def test_case_1_should_have_balanced_inputs(self):
    case = Case1()
    data = case.get_clean_data(count=100)
    outputs = data.get_correct_outputs()

    output_encoder = case.get_tracr_model().output_encoder
    encoded_outputs: List[List[int]] = [output_encoder.encode(o[3:]) for o in outputs]

    # assert we have 20% outputs of all 1s, 20% all 0s, 60% mixed
    assert len([o for o in encoded_outputs if o.count(0) == len(o)]) == 15
    assert len([o for o in encoded_outputs if o.count(1) == len(o)]) == 15
    assert len([o for o in encoded_outputs if o.count(0) != len(o) and o.count(1) != len(o)]) == 70

  def test_case_5_should_have_balanced_inputs(self):
    case = Case5()
    data = case.get_clean_data(count=100)
    outputs = data.get_correct_outputs()

    output_encoder = case.get_tracr_model().output_encoder
    encoded_outputs = [str(output_encoder.encode(o[1:])) for o in outputs]
    different_outputs = set(encoded_outputs)

    # assert each different output is represented in the data
    for output in different_outputs:
      assert encoded_outputs.count(output) >= 45

