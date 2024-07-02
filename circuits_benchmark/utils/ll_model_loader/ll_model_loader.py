from typing import Optional, Tuple

from iit.utils.correspondence import Correspondence
from transformer_lens import HookedTransformer

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase


class LLModelLoader(object):
  def __init__(self, case: BenchmarkCase):
    self.case = case

  def load_ll_model_and_correspondence(
      self,
      device: str,
      output_dir: Optional[str] = None,
      same_size: bool = False,
      *args, **kwargs
  ) -> Tuple[Correspondence, HookedTransformer]:
    raise NotImplementedError()

  def get_output_suffix(self):
    raise NotImplementedError()