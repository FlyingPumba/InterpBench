import random
from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp

from circuits_benchmark.benchmark.vocabs import TRACR_PAD, TRACR_BOS
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput


class Case109(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_check_last_two_equal()

  def get_task_description(self) -> str:
    return "Check if the last two elements in the input list are equal."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False

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

    sorted_vocab = list(sorted(self.get_vocab()))

    # Generate 'true' data: sequences where the last two elements are equal
    for _ in range(true_data_count):
      seq_len = random.randint(min_seq_len, max_seq_len)
      random_sequence = random.choices(sorted_vocab, k=seq_len - 3)
      # Ensure last two elements are equal
      last_element = random.choice(sorted_vocab)
      balanced_input = random_sequence + [last_element, last_element]

      pad_len = max_seq_len - seq_len
      pad = [TRACR_PAD] * pad_len

      output = self.get_correct_output_for_input(balanced_input)

      input_data.append([TRACR_BOS] + balanced_input + pad)
      output_data.append([TRACR_BOS] + output + pad)

    # Generate 'false' data: sequences where the last two elements are different
    for _ in range(false_data_count):
      seq_len = random.randint(min_seq_len, max_seq_len)
      random_sequence = random.choices(sorted_vocab, k=seq_len - 3)
      # Ensure last two elements are different
      last_element_1 = random.choice(sorted_vocab)
      # Make sure the second-to-last element is different from the last element
      last_element_2 = random.choice([x for x in sorted_vocab if x != last_element_1])
      unbalanced_input = random_sequence + [last_element_1, last_element_2]

      pad_len = max_seq_len - seq_len
      pad = [TRACR_PAD] * pad_len

      output = self.get_correct_output_for_input(unbalanced_input)

      input_data.append([TRACR_BOS] + unbalanced_input + pad)
      output_data.append([TRACR_BOS] + output + pad)

    return input_data, output_data


def make_check_last_two_equal() -> rasp.SOp:
  len = rasp.SelectorWidth(rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE))
  last_idx = rasp.Map(lambda x: x - 1, len)
  second_to_last_idx = rasp.Map(lambda x: x - 2, len)
  last_elt = rasp.Aggregate(rasp.Select(rasp.indices, last_idx, rasp.Comparison.EQ), rasp.tokens)
  second_to_last_elt = rasp.Aggregate(rasp.Select(rasp.indices, second_to_last_idx, rasp.Comparison.EQ), rasp.tokens)
  return rasp.SequenceMap(lambda x, y: 1 if x == y else 0, last_elt, second_to_last_elt)
