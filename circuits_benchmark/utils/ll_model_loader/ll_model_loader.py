from typing import Optional, Tuple

from iit.utils.correspondence import Correspondence

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer


class LLModelLoader(object):
  def __init__(self, case: BenchmarkCase):
    self.case = case

  def load_ll_model_and_correspondence(
      self,
      load_from_wandb: bool,
      device: str,
      output_dir: Optional[str] = None,
      same_size: bool = False,
  ) -> Tuple[Correspondence, HookedTracrTransformer]:
    raise NotImplementedError()

  def get_output_suffix(self):
    raise NotImplementedError()