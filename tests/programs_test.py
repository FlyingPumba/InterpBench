import unittest
from typing import Any, List

import numpy as np
from tracr.rasp import rasp
from tracr.transformer.encoder import CategoricalEncoder

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.cases.case_13 import make_token_trend_analysis
from circuits_benchmark.benchmark.cases.case_28 import make_token_mirroring
from circuits_benchmark.benchmark.cases.case_32 import make_token_boundary_detector
from circuits_benchmark.benchmark.cases.case_38 import make_token_alternation_checker
from circuits_benchmark.benchmark.cases.case_8 import make_token_replacer
from circuits_benchmark.benchmark.common_programs import make_unique_token_extractor, detect_pattern
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from circuits_benchmark.benchmark.vocabs import TRACR_BOS, TRACR_PAD
from circuits_benchmark.utils.get_cases import get_cases


class ProgramsTest(unittest.TestCase):
    def compare_valid_positions(self,
                                expected_output: List[Any],
                                predicted_output: List[Any],
                                is_categorical: bool,
                                atol: float = 1.e-5) -> List[bool]:
        """Return a list of booleans indicating if the predicted output is correct at each position.
        This method only considers the positions that are not "BOS", "PAD", or None in the expected output.
        """
        expected_output, predicted_output = self.remove_invalid_positions(expected_output, predicted_output)
        return self.compare_positions(expected_output, predicted_output, is_categorical, atol)

    def remove_invalid_positions(
        self,
        expected_output: List[Any],
        predicted_output: List[Any]
    ) -> (List[Any], List[Any]):
        """Return the expected and predicted outputs without the invalid positions."""
        assert not isinstance(expected_output[0], list), "expected_output should be a single output"

        # Figure out the indices in expected output that are "BOS", "PAD" or None
        skip_indices = set([i for i, elem in enumerate(expected_output) if elem in [TRACR_BOS, TRACR_PAD, None]])

        # Remove such elements from expected and predicted output
        expected_output = [elem for i, elem in enumerate(expected_output) if i not in skip_indices]
        predicted_output = [elem for i, elem in enumerate(predicted_output) if i not in skip_indices]

        return expected_output, predicted_output

    def compare_positions(self, expected_output, predicted_output, is_categorical, atol):
        if is_categorical:
            correct_positions = [elem1 == elem2 for elem1, elem2 in zip(predicted_output, expected_output)]
        else:
            # compare how close the outputs are numerically without taking into account the BOS or PAD tokens
            correct_positions = np.isclose(expected_output, predicted_output, atol=atol).tolist()
        return correct_positions

    def run_case_tests_on_tracr_model(
        self,
        case: TracrBenchmarkCase,
        data_size: int = 10,
        atol: float = 1.e-2,
        fail_on_error: bool = True
    ) -> float:
        tracr_model = case.get_tracr_output().model
        dataset = case.get_clean_data(max_samples=data_size, encoded_dataset=False)
        inputs = dataset.get_inputs()
        expected_outputs = dataset.get_targets()

        is_categorical = isinstance(tracr_model.output_encoder, CategoricalEncoder)

        correct_count = 0
        for i in range(len(inputs)):
            input = inputs[i]
            expected_output = expected_outputs[i]
            decoded_output = tracr_model.apply(input).decoded
            correct = all(self.compare_valid_positions(expected_output, decoded_output, is_categorical, atol))

            if not correct and fail_on_error:
                raise ValueError(f"Failed test for {self} on tracr model."
                                 f"\n >>> Input: {input}"
                                 f"\n >>> Expected: {expected_output}"
                                 f"\n >>> Got: {decoded_output}")
            elif correct:
                correct_count += 1

        return correct_count / len(inputs)

    def run_case_tests_on_ll_model(
        self,
        case: TracrBenchmarkCase,
        data_size: int = 10,
        atol: float = 1.e-2,
        fail_on_error: bool = True
    ) -> float:
        hl_model = case.get_hl_model()

        dataset = case.get_clean_data(max_samples=data_size, encoded_dataset=False)
        inputs = dataset.get_inputs()
        expected_outputs = dataset.get_targets()
        decoded_outputs = hl_model(inputs, return_type="decoded")

        correct_count = 0
        for i in range(len(expected_outputs)):
            input = inputs[i]
            expected_output = expected_outputs[i]
            decoded_output = decoded_outputs[i]
            correct = all(self.compare_valid_positions(expected_output, decoded_output, case.is_categorical(), atol))

            if not correct and fail_on_error:
                raise ValueError(f"Failed test for {self} on tl model."
                                 f"\n >>> Input: {input}"
                                 f"\n >>> Expected: {expected_output}"
                                 f"\n >>> Got: {decoded_output}")
            elif correct:
                correct_count += 1

        return correct_count / len(expected_outputs)

    def test_all_cases_can_be_compiled_and_have_expected_outputs(self):
        for case in get_cases():
            if not isinstance(case, TracrBenchmarkCase):
                print(f"Skipping {case} because it is not a TracrBenchmarkCase")
                continue

            print(f"\nCompiling {case}")
            self.run_case_tests_on_tracr_model(case)
            self.run_case_tests_on_ll_model(case)

            # print some stats about the model
            ll_model = case.get_ll_model()
            print(f"Is categorical status: {case.is_categorical()}")
            print(f"Number of layers: {len(ll_model.blocks)}")
            max_heads = max([len(b.attn.W_Q) for b in ll_model.blocks])
            print(f"Max number of attention heads: {max_heads}")

    def test_make_unique_token_extractor(self):
        program = make_unique_token_extractor(rasp.tokens)

        assert program(["the", "quick", "brown", "fox"]) == ["the", "quick", "brown", "fox"]
        assert program(["the", "quick", "brown", "the"]) == ["the", "quick", "brown", None]
        assert program(["the", "quick", "brown", "quick"]) == ["the", "quick", "brown", None]
        assert program(["the", "quick", "brown", "brown"]) == ["the", "quick", "brown", None]
        assert program(["the", "quick", "brown", "the", "quick", "brown"]) == ["the", "quick", "brown", None, None,
                                                                               None]

    def test_make_token_mirroring(self):
        program = make_token_mirroring(rasp.tokens)

        assert program(["abc", "def", "ghi"]) == ["cba", "fed", "ihg"]

    def test_make_token_boundary_detector(self):
        program = make_token_boundary_detector(rasp.tokens)

        assert program(["apple", "banana", "apple", "orange"]) == [None, True, True, True]
        assert program(["apple", "banana", "banana", "orange"]) == [None, True, False, True]
        assert program(["apple", "apple", "banana", "orange"]) == [None, False, True, True]

    def test_detect_pattern(self):
        program = detect_pattern(rasp.tokens, "abc")

        assert program("abcabc") == [None, None, True, False, False, True]
        assert program("abcab") == [None, None, True, False, False]
        assert program("ab") == [None, None]
        assert program("abc") == [None, None, True]
        assert program("abca") == [None, None, True, False]
        assert program("cabca") == [None, None, False, True, False]
        assert program("aaaaa") == [None, None, False, False, False]

    def test_make_token_alternation_checker(self):
        program = make_token_alternation_checker(rasp.tokens)
        assert program(["cat", "dog", "cat", "dog"]) == [None, True, True, None]
        assert program(["cat", "dog", "cat", "cat"]) == [None, True, False, None]
        assert program(["dog", "dog", "cat", "dog"]) == [None, False, True, None]
        assert program(["cat", "dog", "dog", "dog"]) == [None, False, False, None]
        assert program(["cat", "cat", "dog", "dog"]) == [None, False, False, None]
        assert program(["cat", "cat", "cat", "cat"]) == [None, False, False, None]

    def test_make_token_replacer(self):
        program = make_token_replacer(rasp.tokens, "findme", "-")
        vocab = list(vocabs.get_words_vocab())

        assert program([vocab[0], "findme", vocab[1]]) == [vocab[0], "-", vocab[1]]

    def test_make_token_trend_analysis(self):
        program = make_token_trend_analysis(rasp.tokens)

        assert program([1, 2, 3, 3, 2, 1]) == ["increasing", "increasing", "constant", "decreasing", "decreasing", None]
        assert program([1, 1, 1, 1, 1, 1]) == ["constant", "constant", "constant", "constant", "constant", None]
        assert program([1, 2, 3, 4, 5, 6]) == ["increasing", "increasing", "increasing", "increasing", "increasing", None]
        assert program([6, 5, 4, 3, 2, 1]) == ["decreasing", "decreasing", "decreasing", "decreasing", "decreasing", None]
        assert program([1, 2, 3, 2, 1, 2]) == ["increasing", "increasing", "decreasing", "decreasing", "increasing", None]
        assert program([1, 2, 3, 2, 3, 2]) == ["increasing", "increasing", "decreasing", "increasing", "decreasing", None]
        assert program([2, 0, 0, 0, 1, 0, 1, 1, 0]) == ['decreasing', 'constant', 'constant', 'increasing', 'decreasing', 'increasing', 'constant', 'decreasing', None]
