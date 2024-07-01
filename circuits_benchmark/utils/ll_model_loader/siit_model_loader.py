import pickle
from typing import Optional, Tuple

from iit.utils.correspondence import Correspondence

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from circuits_benchmark.utils.ll_model_loader.best_weights import get_best_weight
from circuits_benchmark.utils.iit.wandb_loader import load_model_from_wandb
from circuits_benchmark.utils.ll_model_loader.ll_model_loader import LLModelLoader


class SIITModelLoader(LLModelLoader):
  def __init__(self, case: BenchmarkCase, weights: str | None = None):
    super().__init__(case)
    self.weights = weights

    if self.weights is None or self.weights == "best":
      print(f"Getting best weights for {self.case.get_name()}")
      self.weights = get_best_weight(self.case.get_name())

  def get_output_suffix(self) -> str:
    return self.__str__()

  def __repr__(self) -> str:
    return self.__str__()

  def __str__(self) -> str:
    return f"siit_weights_{self.weights}"

  def load_ll_model_and_correspondence(
      self,
      load_from_wandb: bool,
      device: str,
      output_dir: Optional[str] = None,
      same_size: bool = False,
  ) -> Tuple[Correspondence, HookedTracrTransformer]:
    hl_model = self.case.get_hl_model(device=device)
    try:
      ll_cfg = pickle.load(
        open(
          f"{output_dir}/ll_models/{self.case.get_name()}/ll_model_cfg_{self.weights}.pkl",
          "rb",
        )
      )
    except FileNotFoundError:
      ll_cfg = self.case.get_ll_model_cfg(same_size=same_size)

    ll_model = HookedTracrTransformer(
      ll_cfg,
      hl_model.tracr_input_encoder,
      hl_model.tracr_output_encoder,
      hl_model.residual_stream_labels,
    )

    hl_ll_corr = self.case.get_correspondence()

    if load_from_wandb:
      try:
        load_model_from_wandb(
          self.case.get_name(), self.weights, output_dir, same_size=same_size
        )
      except FileNotFoundError:
        raise FileNotFoundError(
          f"Could not find SIIT model with weights {self.weights} for case {self.case.get_name()} in wandb"
        )
    ll_model.load_weights_from_file(
      f"{output_dir}/ll_models/{self.case.get_name()}/ll_model_{self.weights}.pth"
    )
    ll_model.to(device)

    return hl_ll_corr, ll_model
