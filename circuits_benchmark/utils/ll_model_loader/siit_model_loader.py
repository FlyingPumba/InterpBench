import pickle
import tempfile
from typing import Optional, Tuple

import torch
from transformer_lens import HookedTransformer

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.iit.wandb_loader import load_model_from_wandb
from circuits_benchmark.utils.ll_model_loader.best_weights import get_best_weight
from circuits_benchmark.utils.ll_model_loader.ll_model_loader import LLModelLoader
from iit.utils.correspondence import Correspondence


class SIITModelLoader(LLModelLoader):
  def __init__(self,
               case: BenchmarkCase,
               weights: str | None = None,
               load_from_wandb: bool | None = False):
    super().__init__(case)
    self.weights = weights
    self.load_from_wandb = load_from_wandb

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
      device: str,
      output_dir: Optional[str] = None,
      same_size: bool = False,
      *args, **kwargs
  ) -> Tuple[Correspondence, HookedTransformer]:
    assert self.load_from_wandb or output_dir is not None, ValueError("Either load from wandb or provide directory to load from")
    if output_dir is None:
      # make temporary directory
      with tempfile.TemporaryDirectory() as output_dir:
        print(f"Created temporary directory at {output_dir}")
        return self.load_ll_model_and_correspondence(
          device, output_dir, same_size=same_size, *args, **kwargs
        )

    if self.load_from_wandb:
      try:
        load_model_from_wandb(
          self.case.get_name(), self.weights, output_dir, same_size=same_size
        )
      except FileNotFoundError:
        raise FileNotFoundError(
          f"Could not find SIIT model with weights {self.weights} for case {self.case.get_name()} in wandb"
        )
      
    try:
      ll_cfg = pickle.load(
        open(
          f"{output_dir}/ll_models/{self.case.get_name()}/ll_model_cfg_{self.weights}.pkl",
          "rb",
        )
      )
    except FileNotFoundError:
      ll_cfg = self.case.get_ll_model_cfg(same_size=same_size, *args, **kwargs)
    ll_cfg["device"] = device
    ll_model = HookedTransformer(ll_cfg)
    hl_ll_corr = self.case.get_correspondence(same_size=same_size, *args, **kwargs)

    ll_model.load_state_dict(torch.load(
      f"{output_dir}/ll_models/{self.case.get_name()}/ll_model_{self.weights}.pth",
      map_location=device))

    return hl_ll_corr, ll_model
