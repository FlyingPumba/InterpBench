import unittest
from typing import List

import pytest

from circuits_benchmark.benchmark.cases.case_1 import Case1
from circuits_benchmark.benchmark.cases.case_3 import Case3


class TestGetCleanData:
  def test_get_all_clean_data(self):
    case = Case3()
    data = case.get_clean_data(max_samples=None, variable_length_seqs=True)

    expected_total_data_len = 320
    assert len(data.get_inputs()) == expected_total_data_len
    assert case.get_total_data_len() == expected_total_data_len

  def test_get_partial_clean_data(self):
    case = Case3()
    data = case.get_clean_data(max_samples=10, variable_length_seqs=True)
    assert len(data.get_inputs()) == 10

  def test_case_1_should_have_balanced_inputs(self):
    case = Case1()
    data = case.get_clean_data(max_samples=100, encoded_dataset=False)
    outputs = data.get_targets()

    output_encoder = case.get_hl_model().tracr_output_encoder
    encoded_outputs: List[List[int]] = [output_encoder.encode(o[3:]) for o in outputs]

    # assert we have 20% outputs of all 1s, 20% all 0s, 60% mixed
    assert len([o for o in encoded_outputs if o.count(0) == len(o)]) == 15
    assert len([o for o in encoded_outputs if o.count(1) == len(o)]) == 15
    assert len([o for o in encoded_outputs if o.count(0) != len(o) and o.count(1) != len(o)]) == 70
