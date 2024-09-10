import random
from typing import Set, List

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.common_programs import make_reverse
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp

from circuits_benchmark.benchmark.vocabs import TRACR_BOS, TRACR_PAD
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput


class Case17(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_palindrome_detection(rasp.tokens)

  def get_task_description(self) -> str:
    return "Detect if input sequence is a palindrome."

  def supports_causal_masking(self) -> bool:
    return False

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=10)

  def get_true_input(self, length: int) -> List[str]:
    # Generate a random sequence and make it a palindrome
    half_length = (length + 1) // 2
    vocab = sorted(self.get_vocab())
    first_half = random.choices(vocab, k=half_length)
    palindrome_sequence = first_half + first_half[::-1] if length % 2 == 0 else first_half + first_half[-2::-1]
    return palindrome_sequence

  def get_false_input(self, length: int) -> List[str]:
    # Generate a random non-palindromic sequence
    vocab = sorted(self.get_vocab())
    non_palindrome = random.choices(vocab, k=length)
    # Ensure it is not a palindrome
    if non_palindrome == non_palindrome[::-1]:
      # Force a change to break the palindrome
      non_palindrome[length // 2] = random.choice([x for x in vocab if x != non_palindrome[length // 2]])
    return non_palindrome

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

    # Generate palindromic data
    for _ in range(true_data_count):
      seq_len = random.randint(min_seq_len, max_seq_len)
      palindromic_input = self.get_true_input(seq_len - 1)

      pad_len = max_seq_len - seq_len
      pad = [TRACR_PAD] * pad_len

      output = self.get_correct_output_for_input(palindromic_input)

      input_data.append([TRACR_BOS] + palindromic_input + pad)
      output_data.append([TRACR_BOS] + output + pad)

    # Generate non-palindromic data
    for _ in range(false_data_count):
      seq_len = random.randint(min_seq_len, max_seq_len)
      non_palindromic_input = self.get_false_input(seq_len - 1)

      pad_len = max_seq_len - seq_len
      pad = [TRACR_PAD] * pad_len

      output = self.get_correct_output_for_input(non_palindromic_input)

      input_data.append([TRACR_BOS] + non_palindromic_input + pad)
      output_data.append([TRACR_BOS] + output + pad)

    return input_data, output_data


def make_palindrome_detection(sop: rasp.SOp) -> rasp.SOp:
    """
    Detects palindromes in a sequence of characters.

    Example usage:
      palindrome_detect = make_palindrome_detection(rasp.tokens)
      palindrome_detect("racecar")
      >> [False, False, False, True, False, False, False]

    Args:
      sop: SOp representing the sequence to analyze.

    Returns:
      A SOp that maps an input sequence to a boolean sequence, where True 
      indicates a palindrome at that position.
    """
    reversed_sop = make_reverse(sop)
    palindrome_sop = rasp.SequenceMap(
        lambda x, y: x == y, sop, reversed_sop).named("palindrome_detection")
    return palindrome_sop
