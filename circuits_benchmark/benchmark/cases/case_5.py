import random
from typing import Set, List

from tracr.rasp import rasp

from circuits_benchmark.benchmark.common_programs import make_shuffle_dyck
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from circuits_benchmark.benchmark.vocabs import TRACR_BOS, TRACR_PAD
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput


class Case5(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_shuffle_dyck(pairs=["()", "{}"])

  def get_task_description(self) -> str:
    return "Returns 1 if a set of parentheses are balanced, 0 else."

  def get_vocab(self) -> Set:
    return {"(", ")", "{", "}", "x"}

  def supports_causal_masking(self) -> bool:
    return False

  def gen_balanced_input(self, length) -> List[str]:
    open_brackets = ['(', '{']
    close_brackets = [')', '}']

    brackets_map = {
      '(': ')',
      ')': '(',
      '{': '}',
      '}': '{'
    }

    input = []
    open_brackets_stack = []

    while len(input) < length:
      # let's figure out which characters we can use at this position
      chars = set()

      if len(open_brackets_stack) > 0:
        # we can definitely use closed brackets
        for char in open_brackets_stack:
          chars.add(brackets_map[char])

      # if the space left is less or equal than as the number of opened brackets plus one, we can also use the 'x' character
      if len(open_brackets_stack) + 1 <= length - len(input):
        chars.add('x')

      # if the space left is less or equal than as the number of opened brackets plus two, we can also use open brackets
      if len(open_brackets_stack) + 2 <= length - len(input):
        chars.update(open_brackets)

      # choose a random character from the valid set
      char = random.choice(list(chars))
      input.append(char)

      if char in open_brackets:
        open_brackets_stack.append(char)
      elif char in close_brackets:
        # remove the corresponding open bracket
        open_brackets_stack.remove(brackets_map[char])

    return input

  def sample_data(self, count, min_seq_len, max_seq_len) -> (HookedTracrTransformerBatchInput, HookedTracrTransformerBatchInput):
    """Samples random data for this benchmark case, making sure that we get half of the data with balanced parentheses/brakets and half with unbalanced ones."""
    input_data: HookedTracrTransformerBatchInput = []
    output_data: HookedTracrTransformerBatchInput = []

    # we use self.get_correct_output_for_input(sample) to get the correct output for each input
    balanced_data_count = count // 2
    unbalanced_data_count = count - balanced_data_count

    # generate balanced data: equal number of open and close parentheses/brakets in each sample
    for _ in range(balanced_data_count):
      seq_len = random.randint(min_seq_len, max_seq_len)
      balanced_input = self.gen_balanced_input(seq_len-1)

      pad_len = max_seq_len - seq_len
      pad = [TRACR_PAD] * pad_len

      output = self.get_correct_output_for_input(balanced_input)

      input_data.append([TRACR_BOS] + balanced_input + pad)
      output_data.append([TRACR_BOS] + output + pad)

    sorted_vocab = ['(', ')', 'x', '{', '}']
    for _ in range(unbalanced_data_count):
      input, output = self.gen_random_input_output(sorted_vocab, min_seq_len, max_seq_len)
      input_data.append(input)
      output_data.append(output)

    return input_data, output_data