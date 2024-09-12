import random
from typing import Set, List

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from circuits_benchmark.benchmark.vocabs import TRACR_BOS, TRACR_PAD
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput


class Case125(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_check_alternating()

  def get_task_description(self) -> str:
    return "Check whether a sequence consists of alternating even and odd elements."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False

  def get_true_input(self, length: int) -> List[str]:
    # Generate a sequence that alternates between even and odd numbers
    sorted_vocab = sorted(self.get_vocab())
    even_numbers = [x for x in sorted_vocab if x % 2 == 0]
    odd_numbers = [x for x in sorted_vocab if x % 2 == 1]

    # Create alternating sequence starting with an even number
    alternating_sequence = []
    for i in range(length):
      if i % 2 == 0:
        alternating_sequence.append(random.choice(even_numbers))
      else:
        alternating_sequence.append(random.choice(odd_numbers))

    return alternating_sequence

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

    # Generate balanced data: sequences that alternate between even and odd elements
    for _ in range(true_data_count):
      seq_len = random.randint(min_seq_len, max_seq_len)
      balanced_input = self.get_true_input(seq_len - 1)

      pad_len = max_seq_len - seq_len
      pad = [TRACR_PAD] * pad_len

      output = self.get_correct_output_for_input(balanced_input)

      input_data.append([TRACR_BOS] + balanced_input + pad)
      output_data.append([TRACR_BOS] + output + pad)

    # Generate random sequences that do not alternate between even and odd elements
    sorted_vocab = list(sorted(self.get_vocab()))
    for _ in range(false_data_count):
      input, output = self.gen_random_input_output(sorted_vocab, min_seq_len, max_seq_len)
      input_data.append(input)
      output_data.append(output)

    return input_data, output_data


def make_check_alternating() -> rasp.SOp:
  shifted_sequence = rasp.Aggregate(
    rasp.Select(rasp.indices, rasp.indices, lambda k, q: q == k + 1 or k == 1 and q == 0), rasp.tokens)
  shifted_even_odd = rasp.Map(lambda x: 1 if x % 2 == 0 else 0, shifted_sequence)
  even_odd = rasp.Map(lambda x: 1 if x % 2 == 1 else 0, rasp.tokens)
  checks = rasp.SequenceMap(lambda x, y: 1 if x == y else 0, shifted_even_odd, even_odd)
  zero_selector = rasp.Select(checks, rasp.Map(lambda x: 0, rasp.indices), rasp.Comparison.EQ)
  return rasp.Map(lambda x: 0 if x > 0 else 1, rasp.SelectorWidth(zero_selector))
