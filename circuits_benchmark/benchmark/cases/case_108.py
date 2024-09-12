import random
from typing import Set, List

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from circuits_benchmark.benchmark.vocabs import TRACR_PAD, TRACR_BOS
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput


class Case108(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_test_at_least_two_equal()

    def get_task_description(self) -> str:
        return "Check if at least two elements in the input list are equal."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab()

    def supports_causal_masking(self) -> bool:
        return False

    def get_true_input(self, length: int) -> List[str]:
        """
        Generate an input list with at least two elements being equal.
        """
        sorted_vocab = sorted(self.get_vocab())
        # Ensure at least two elements are the same
        if length < 2:
            raise ValueError("Length must be at least 2 for true input generation.")

        # Select `length - 1` unique elements and repeat one element to ensure at least two are the same
        unique_elements = random.sample(sorted_vocab, length - 1)
        duplicate_element = random.choice(unique_elements)
        random_sequence = unique_elements + [duplicate_element]
        random.shuffle(random_sequence)  # Shuffle to avoid predictable order

        return random_sequence

    def get_false_input(self, length: int) -> List[str]:
        """
        Generate an input list where all elements are unique.
        """
        sorted_vocab = sorted(self.get_vocab())
        # Sample `length` unique elements
        random_sequence = random.sample(sorted_vocab, length)
        return random_sequence

    def sample_data(
        self,
        count: int,
        min_seq_len: int,
        max_seq_len: int
    ) -> (HookedTracrTransformerBatchInput, HookedTracrTransformerBatchInput):
        input_data: HookedTracrTransformerBatchInput = []
        output_data: HookedTracrTransformerBatchInput = []

        true_data_count = count // 2
        false_data_count = count - true_data_count

        # Generate balanced data with true cases
        for _ in range(true_data_count):
            seq_len = random.randint(min_seq_len, max_seq_len)
            true_input = self.get_true_input(seq_len - 1)

            pad_len = max_seq_len - seq_len
            pad = [TRACR_PAD] * pad_len

            output = self.get_correct_output_for_input(true_input)

            input_data.append([TRACR_BOS] + true_input + pad)
            output_data.append([TRACR_BOS] + output + pad)

        # Generate balanced data with false cases
        for _ in range(false_data_count):
            seq_len = random.randint(min_seq_len, max_seq_len)
            false_input = self.get_false_input(seq_len - 1)

            pad_len = max_seq_len - seq_len
            pad = [TRACR_PAD] * pad_len

            output = self.get_correct_output_for_input(false_input)

            input_data.append([TRACR_BOS] + false_input + pad)
            output_data.append([TRACR_BOS] + output + pad)

        return input_data, output_data


def make_test_at_least_two_equal() -> rasp.SOp:
    equal_selector = rasp.Select(rasp.tokens, rasp.tokens, lambda x, y: x == y)
    checks = rasp.SelectorWidth(equal_selector)
    greater_than_2 = rasp.Select(checks, rasp.Map(lambda x: 2, rasp.indices), rasp.Comparison.GEQ)
    return rasp.Map(lambda x: 1 if x > 0 else 0, rasp.SelectorWidth(greater_than_2))
