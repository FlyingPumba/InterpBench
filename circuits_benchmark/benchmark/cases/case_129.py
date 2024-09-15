import random
from typing import Set, List

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from circuits_benchmark.benchmark.vocabs import TRACR_PAD, TRACR_BOS
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput


class Case129(TracrBenchmarkCase):
    def get_program(self) -> rasp.SOp:
        return make_check_multiple_of_n()

    def get_task_description(self) -> str:
        return "Checks if all elements are a multiple of n (set the default at 2)."

    def get_vocab(self) -> Set:
        return vocabs.get_int_numbers_vocab(max=30)

    def supports_causal_masking(self) -> bool:
        return False

    def get_multiple_of_n_input(self, length: int, n: int = 2) -> List[str]:
        # Get the sorted list of integers from the vocabulary that are multiples of n
        sorted_vocab = sorted(self.get_vocab())
        multiples_of_n = [x for x in sorted_vocab if x % n == 0]
        # Sample `length` elements randomly from the multiples
        random_sequence = random.sample(multiples_of_n, length)
        return random_sequence

    def get_not_multiple_of_n_input(self, length: int, n: int = 2) -> List[str]:
        # Generate a random sequence that contains at least one element not a multiple of n
        sorted_vocab = sorted(self.get_vocab())
        multiples_of_n = [x for x in sorted_vocab if x % n == 0]
        non_multiples_of_n = [x for x in sorted_vocab if x % n != 0]

        # Ensure at least one non-multiple of `n`
        sequence = random.sample(multiples_of_n, length - 1)
        sequence.append(random.choice(non_multiples_of_n))
        random.shuffle(sequence)  # Shuffle to avoid predictable patterns

        return sequence

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

        # Generate balanced data: sequences with all elements as multiples of `n`
        for _ in range(true_data_count):
            seq_len = random.randint(min_seq_len, max_seq_len)
            multiple_input = self.get_multiple_of_n_input(seq_len - 1)

            pad_len = max_seq_len - seq_len
            pad = [TRACR_PAD] * pad_len

            output = self.get_correct_output_for_input(multiple_input)

            input_data.append([TRACR_BOS] + multiple_input + pad)
            output_data.append([TRACR_BOS] + output + pad)

        # Generate balanced data: sequences with at least one element not a multiple of `n`
        for _ in range(false_data_count):
            seq_len = random.randint(min_seq_len, max_seq_len)
            not_multiple_input = self.get_not_multiple_of_n_input(seq_len - 1)

            pad_len = max_seq_len - seq_len
            pad = [TRACR_PAD] * pad_len

            output = self.get_correct_output_for_input(not_multiple_input)

            input_data.append([TRACR_BOS] + not_multiple_input + pad)
            output_data.append([TRACR_BOS] + output + pad)

        return input_data, output_data


def make_check_multiple_of_n(n=2) -> rasp.SOp:
    checks = rasp.Map(lambda x: 1 if x % n == 0 else 0, rasp.tokens).named(f"check_multiple_of_{n}")
    zero_selector = rasp.Select(checks, rasp.Map(lambda x: 0, rasp.indices), rasp.Comparison.EQ)
    return rasp.Map(lambda x: 0 if x > 0 else 1, rasp.SelectorWidth(zero_selector))
