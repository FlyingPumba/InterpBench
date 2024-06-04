import unittest

from circuits_benchmark.benchmark.cases.case_16 import make_lexical_density_calculator
from circuits_benchmark.benchmark.cases.case_27 import make_token_positional_balance_analyzer
from circuits_benchmark.benchmark.cases.case_28 import make_token_mirroring
from circuits_benchmark.benchmark.cases.case_32 import make_token_boundary_detector
from circuits_benchmark.benchmark.cases.case_38 import make_token_alternation_checker
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

  def test_make_token_alternation_checker(self):
    program = make_token_alternation_checker(rasp.tokens)

    assert program(["cat", "dog", "cat", "dog"]) == [None, True, True, None]
    assert program(["cat", "dog", "cat", "cat"]) == [None, True, False, None]
    assert program(["dog", "dog", "cat", "dog"]) == [None, False, True, None]
    assert program(["cat", "dog", "dog", "dog"]) == [None, False, False, None]
    assert program(["cat", "cat", "dog", "dog"]) == [None, False, False, None]
    assert program(["cat", "cat", "cat", "cat"]) == [None, False, False, None]

  def test_make_token_positional_balance_analyzer(self):
    program = make_token_positional_balance_analyzer(rasp.tokens)

    assert program(["a", "b", "c", "d", "e"]) == ["front", "front", "center", "center", "rear"]

  def test_make_token_mirroring(self):
    program = make_token_mirroring(rasp.tokens)

    assert program(["abc", "def", "ghi"]) == ["cba", "fed", "ihg"]

  def test_make_token_boundary_detector(self):
    program = make_token_boundary_detector(rasp.tokens)

    assert program(["apple", "banana", "apple", "orange"]) == [None, True, True, True]
    assert program(["apple", "banana", "banana", "orange"]) == [None, True, False, True]
    assert program(["apple", "apple", "banana", "orange"]) == [None, False, True, True]
