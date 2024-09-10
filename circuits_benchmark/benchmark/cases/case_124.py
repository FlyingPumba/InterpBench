import random
from typing import Set, Sequence, List

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp

from circuits_benchmark.benchmark.vocabs import TRACR_PAD, TRACR_BOS
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput


class Case124(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_check_all_equal()

  def get_task_description(self) -> str:
    return "Check if all elements in a list are equal."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False

  def get_true_input(self, length: int) -> List[str]:
    # Randomly select a single element from the vocabulary
    vocab_list = list(self.get_vocab())
    element = random.choice(vocab_list)
    # Repeat the element `length` times to ensure all elements are equal
    return [element] * length

  def get_false_input(self, length: int) -> List[str]:
    # Randomly select multiple different elements from the vocabulary
    vocab_list = list(self.get_vocab())
    # Ensure we pick at least 2 different elements
    if length < 2:
      raise ValueError("Length must be at least 2 to create a false input.")
    # Sample different elements to ensure not all are the same
    elements = random.sample(vocab_list, k=min(length, len(vocab_list)))
    # Extend the list to the required length with repeats if necessary
    return (elements * (length // len(elements) + 1))[:length]

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

    # Generate balanced data: sequences where all elements are equal (True cases)
    for _ in range(true_data_count):
      seq_len = random.randint(min_seq_len, max_seq_len)
      equal_input = self.get_true_input(seq_len - 1)

      pad_len = max_seq_len - seq_len
      pad = [TRACR_PAD] * pad_len

      output = self.get_correct_output_for_input(equal_input)

      input_data.append([TRACR_BOS] + equal_input + pad)
      output_data.append([TRACR_BOS] + output + pad)

    # Generate sequences where not all elements are equal (False cases)
    for _ in range(false_data_count):
      seq_len = random.randint(min_seq_len, max_seq_len)
      unequal_input = self.get_false_input(seq_len - 1)

      pad_len = max_seq_len - seq_len
      pad = [TRACR_PAD] * pad_len

      output = self.get_correct_output_for_input(unequal_input)

      input_data.append([TRACR_BOS] + unequal_input + pad)
      output_data.append([TRACR_BOS] + output + pad)

    return input_data, output_data


def make_check_all_equal() -> rasp.SOp:
  unequal_selector = rasp.Select(rasp.tokens, rasp.tokens, lambda x, y: x != y)
  checks = rasp.SelectorWidth(unequal_selector)
  zero_selector = rasp.Select(checks, rasp.Map(lambda x: 0, rasp.indices), rasp.Comparison.EQ)
  return rasp.Map(lambda x: 1 if x > 0 else 0, rasp.SelectorWidth(zero_selector))
