import pickle
from typing import Optional, Tuple

from huggingface_hub import hf_hub_download
from iit.utils.correspondence import Correspondence
from transformer_lens import HookedTransformerConfig, HookedTransformer

from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from circuits_benchmark.utils.ll_model_loader.ll_model_loader import LLModelLoader


class InterpBenchModelLoader(LLModelLoader):
  def get_output_suffix(self) -> str:
    return self.__str__()

  def __repr__(self) -> str:
    return self.__str__()

  def __str__(self) -> str:
    return f"interp_bench"

  def load_ll_model_and_correspondence(
      self,
      load_from_wandb: bool,
      device: str,
      output_dir: Optional[str] = None,
      same_size: bool = False,
      *args, **kwargs
  ) -> Tuple[Correspondence, HookedTransformer]:
    assert not same_size, "InterpBench models are never same size"
    assert not load_from_wandb, "InterpBench models cannot loaded from wandb"

    case_name = self.case.get_name()
    model_file = hf_hub_download("cybershiptrooper/InterpBench", subfolder=case_name, filename="ll_model.pth")
    cfg_file = hf_hub_download("cybershiptrooper/InterpBench", subfolder=case_name, filename="ll_model_cfg.pkl")

    hl_model = self.case.get_hl_model()
    try:
      cfg_dict = pickle.load(open(cfg_file, "rb"))
      cfg = HookedTransformerConfig.from_dict(cfg_dict)
      cfg.device = device
      ll_model = HookedTracrTransformer(
        cfg,
        hl_model.tracr_input_encoder,
        hl_model.tracr_output_encoder,
        hl_model.residual_stream_labels,
      )
      ll_model.load_weights_from_file(model_file)
    except FileNotFoundError:
      raise FileNotFoundError(
        f"Could not find InterpBench model for case {self.case.get_name()}"
      )

    hl_ll_corr = self.case.get_correspondence()
    return hl_ll_corr, ll_model