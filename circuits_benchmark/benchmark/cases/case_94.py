import random
from typing import Set, Sequence, List

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp

from circuits_benchmark.benchmark.vocabs import TRACR_PAD, TRACR_BOS
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput


class Case94(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_check_descending()

  def get_task_description(self) -> str:
    return "Check if the sequence is descending"

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False

  def get_true_input(self, length) -> List[str]:
    """Return a random list that is sorted in increasing order"""
    # Get the sorted list of integers from the vocabulary
    sorted_vocab = sorted(self.get_vocab())
    # Sample `length` elements randomly from the sorted_vocab
    random_sequence = random.sample(sorted_vocab, length)
    # Sort the sampled sequence to ensure it is in decreasing order
    return list(sorted(random_sequence, reverse=True))

  def sample_data(
      self,
      count,
      min_seq_len,
      max_seq_len
  ) -> (HookedTracrTransformerBatchInput, HookedTracrTransformerBatchInput):
    input_data: HookedTracrTransformerBatchInput = []
    output_data: HookedTracrTransformerBatchInput = []

    true_data_count = count // 2
    false_data_count = count - true_data_count

    # generate balanced data: equal number of open and close parentheses/brakets in each sample
    for _ in range(true_data_count):
      seq_len = random.randint(min_seq_len, max_seq_len)
      balanced_input = self.get_true_input(seq_len-1)

      pad_len = max_seq_len - seq_len
      pad = [TRACR_PAD] * pad_len

      output = self.get_correct_output_for_input(balanced_input)

      input_data.append([TRACR_BOS] + balanced_input + pad)
      output_data.append([TRACR_BOS] + output + pad)

    sorted_vocab = list(sorted(self.get_vocab()))
    for _ in range(false_data_count):
      input, output = self.gen_random_input_output(sorted_vocab, min_seq_len, max_seq_len)
      input_data.append(input)
      output_data.append(output)

    return input_data, output_data


def make_check_descending():
  shifter = rasp.Select(rasp.indices, rasp.indices, lambda x, y: x == y - 1 or (x == 0 and y == 0))
  shifted = rasp.Aggregate(shifter, rasp.tokens)
  checks = rasp.SequenceMap(lambda x, y: 1 if x >= y else 0, shifted, rasp.tokens)
  zero_selector = rasp.Select(checks, rasp.Map(lambda x: 0, rasp.indices), rasp.Comparison.EQ)
  return rasp.Map(lambda x: 0 if x > 0 else 1, rasp.SelectorWidth(zero_selector))
