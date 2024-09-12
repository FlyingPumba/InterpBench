from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case130(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_clip()

  def get_task_description(self) -> str:
    return "Clips each element to be within a range (make the default range [2, 7])."
    # "Clipping" means that values outside of the range, are turned into the lower or upper bound, whichever is closer.

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab(min=-15, max=15)


def make_clip(min_val=2, max_val=7) -> rasp.SOp:
  # Map all elements to min_val (to use in case of less than min_val)
  all_min_val = rasp.Map(lambda x: min_val, rasp.tokens)
  # Map all elements to max_val (to use in case of greater than max_val)
  all_max_val = rasp.Map(lambda x: max_val, rasp.tokens)

  # Compare each element to min_val and max_val
  less_than_min = rasp.Map(lambda x: x < min_val, rasp.tokens)
  greater_than_max = rasp.Map(lambda x: x > max_val, rasp.tokens)

  # Apply clipping: first, clip to min_val if less than min_val
  clip_min = rasp.SequenceMap(lambda orig, clip: clip if orig < min_val else orig, rasp.tokens, all_min_val)
  # Then, clip to max_val if greater than max_val
  clip_max = rasp.SequenceMap(lambda clipped_min, clip: clip if clipped_min > max_val else clipped_min, clip_min,
                              all_max_val)

  return clip_max
