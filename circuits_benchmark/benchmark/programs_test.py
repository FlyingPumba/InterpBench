import unittest
from typing import List

from circuits_benchmark.benchmark.cases.case_1 import Case1
from circuits_benchmark.benchmark.cases.case_16 import make_lexical_density_calculator
from circuits_benchmark.benchmark.cases.case_3 import Case3
from circuits_benchmark.benchmark.cases.case_5 import Case5
from circuits_benchmark.benchmark.common_programs import make_unique_token_extractor
from tracr.rasp import rasp


class ProgramsTest(unittest.TestCase):
  def test_make_unique_token_extractor(self):
    program = make_unique_token_extractor(rasp.tokens)

    assert program(["the", "quick", "brown", "fox"])  == ["the", "quick", "brown", "fox"]
    assert program(["the", "quick", "brown", "the"]) == ["the", "quick", "brown", None]
    assert program(["the", "quick", "brown", "quick"]) == ["the", "quick", "brown", None]
    assert program(["the", "quick", "brown", "brown"]) == ["the", "quick", "brown", None]
    assert program(["the", "quick", "brown", "the", "quick", "brown"]) == ["the", "quick", "brown", None, None, None]

  def test_make_lexical_density_calculator(self):
    program = make_lexical_density_calculator(rasp.tokens)

    assert program(["the", "quick", "brown", "fox"]) == [1, 1, 1, 1]
    assert program(["the", "quick", "brown", "the"]) == [0.75, 0.75, 0.75, 0]
    assert program(["the", "quick", "the", "the"]) == [0.5, 0.5, 0, 0]
    assert program(["the", "quick", "brown", "the", "the"]) == [0.6, 0.6, 0.6, 0, 0]


