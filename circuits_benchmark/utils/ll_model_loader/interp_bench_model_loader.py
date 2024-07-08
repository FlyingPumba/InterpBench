import os
import pickle
from typing import Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from iit.utils.correspondence import Correspondence
from transformer_lens import HookedTransformerConfig, HookedTransformer

from circuits_benchmark.utils.ll_model_loader.ll_model_loader import LLModelLoader
from circuits_benchmark.utils.project_paths import get_default_output_dir


class InterpBenchModelLoader(LLModelLoader):
  def __init__(self, case):
    super().__init__(case)
    self.cache_dir = os.path.join(get_default_output_dir(), "hf_cache")
    os.makedirs(self.cache_dir, exist_ok=True)

  def get_output_suffix(self) -> str:
    return self.__str__()

  def __repr__(self) -> str:
    return self.__str__()

  def __str__(self) -> str:
    return f"interp_bench"

  def load_ll_model_and_correspondence(
      self,
      device: str,
      output_dir: Optional[str] = None,
      same_size: bool = False,
      *args, **kwargs
  ) -> Tuple[Correspondence, HookedTransformer]:
    assert not same_size, "InterpBench models are never same size"

    case_name = self.case.get_name()
    model_file = hf_hub_download(
      "cybershiptrooper/InterpBench",
      subfolder=case_name,
      filename="ll_model.pth",
      cache_dir=self.cache_dir,
      force_download=False
    )
    cfg_file = hf_hub_download(
      "cybershiptrooper/InterpBench",
      subfolder=case_name,
      filename="ll_model_cfg.pkl",
      cache_dir=self.cache_dir,
      force_download=False
    )

    try:
      cfg_dict = pickle.load(open(cfg_file, "rb"))
      if isinstance(cfg_dict, dict):
        cfg = HookedTransformerConfig.from_dict(cfg_dict)
      else:
        # Some cases in InterpBench have the config as a HookedTransformerConfig object instead of a dict
        assert isinstance(cfg_dict, HookedTransformerConfig)
        cfg = cfg_dict

      cfg.device = device
      if "ioi" in case_name and "eval" in kwargs and kwargs["eval"]:
        # Small hack to enable evaluation mode in the IOI model, that has a different config during training
        cfg.use_hook_mlp_in = True
        cfg.use_attn_result = True
        cfg.use_split_qkv_input = True

      ll_model = HookedTransformer(cfg)
      ll_model.load_state_dict(torch.load(model_file, map_location=device))
    except FileNotFoundError:
      raise FileNotFoundError(
        f"Could not find InterpBench model for case {self.case.get_name()}"
      )

    hl_ll_corr = self.case.get_correspondence(*args, **kwargs)
    return hl_ll_corr, ll_model