import random
from math import ceil, floor
from typing import Set

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.common_programs import shift_by
from circuits_benchmark.benchmark.vocabs import TRACR_PAD, TRACR_BOS
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput
from tracr.rasp import rasp


class Case1(BenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_token_rotation_identifier(rasp.tokens, 2)

  def get_task_description(self) -> str:
    return "Identify if tokens are rotations of each other by a specified number."

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=5)

  def sample_data(self, count, min_seq_len, max_seq_len) -> (HookedTracrTransformerBatchInput, HookedTracrTransformerBatchInput):
    """Samples random data for this benchmark case, making sure that we get balanced data."""
    input_data: HookedTracrTransformerBatchInput = []
    output_data: HookedTracrTransformerBatchInput = []
    sorted_vocab = sorted(self.get_vocab())

    all_true_inputs = ceil(count * 0.15)
    all_false_inputs = ceil(count * 0.15)

    for _ in range(all_true_inputs):
      input, output = self.gen_all_true_input(sorted_vocab, min_seq_len, max_seq_len)
      input_data.append(input)
      output_data.append(output)

    for _ in range(all_false_inputs):
      input, output = self.gen_all_false_input(sorted_vocab, min_seq_len, max_seq_len)
      input_data.append(input)
      output_data.append(output)

    while len(input_data) < count:
      input, output = self.gen_random_input_output(sorted_vocab, min_seq_len, max_seq_len)
      if all(output[2:]) or not any(output[2:]):
        continue

      input_data.append(input)
      output_data.append(output)

    return input_data, output_data

  def gen_all_true_input(self, sorted_vocab, min_seq_len, max_seq_len):
    seq_len_with_bos = random.randint(min_seq_len, max_seq_len)
    seq_len_without_bos = seq_len_with_bos - 1

    sample = random.sample(sorted_vocab, 2)
    sample = sample * floor(seq_len_without_bos / 2)

    # if odd, add the first character again
    if seq_len_without_bos % 2 == 1:
      sample.append(sample[0])

    pad_len = max_seq_len - seq_len_with_bos
    pad = [TRACR_PAD] * pad_len

    output = self.get_correct_output_for_input(sample)
    assert all(output[2:])

    input = [TRACR_BOS] + sample + pad
    output = [TRACR_BOS] + output + pad

    return input, output

  def gen_all_false_input(self, sorted_vocab, min_seq_len, max_seq_len):
    seq_len_with_bos = random.randint(min_seq_len, max_seq_len)
    seq_len_without_bos = seq_len_with_bos - 1

    sample = list(sorted_vocab)
    sample = sample * floor(seq_len_without_bos / len(sorted_vocab))

    # if we have not reached the desired length, add more characters
    if len(sample) < seq_len_without_bos:
      sample.extend(sorted_vocab[:seq_len_without_bos - len(sample)])

    pad_len = max_seq_len - seq_len_with_bos
    pad = [TRACR_PAD] * pad_len

    output = self.get_correct_output_for_input(sample)
    assert all([not o for o in output[2:]])

    input = [TRACR_BOS] + sample + pad
    output = [TRACR_BOS] + output + pad

    return input, output


def make_token_rotation_identifier(sop: rasp.SOp, rotation: int) -> rasp.SOp:
    """
    Identifies if tokens are rotations of each other by a specified number.

    Example usage:
      rotation_identifier = make_token_rotation_identifier(rasp.tokens, 2)
      rotation_identifier(['d', 'e', 'c', 'e', 'e', 'b', 'c', 'c', 'c'])
      >> [None, None, False, True, False, False, False, False, True]
      Because:
        Orig:    ['d', 'e', 'c', 'e', 'e', 'b', 'c', 'c', 'c']
        Shift 2: [_  , _  , 'd', 'e', 'c', 'e', 'e', 'b', 'c']
        Equals:  [_  , _  ,  F ,  T ,  F ,  F ,  F ,  F ,  T ]
    """
    rotated_token = shift_by(rotation, sop)
    rotation_identifier = rasp.SequenceMap(lambda x, y: x == y, sop, rotated_token)
    return rotation_identifier
