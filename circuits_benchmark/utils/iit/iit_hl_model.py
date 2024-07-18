from typing import Any

import torch
from transformer_lens import HookedTransformer


class IITHLModel:
  """A wrapper class to make tracr models compatible with IITModelPair"""

  def __init__(self, hl_model: HookedTransformer, eval_mode: bool = False):
    self.hl_model = hl_model
    self.eval_mode = eval_mode

    for p in hl_model.parameters():
      p.requires_grad = False
      p.to(hl_model.device)

  def __getattr__(self, name: str):
    if hasattr(self.hl_model, name):
      return getattr(self.hl_model, name)
    else:
      raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'"
      )

  def create_hl_output(self, y):
    if self.hl_model.is_categorical():
      y = y.argmax(dim=-1)
      if self.eval_mode:
        y = torch.nn.functional.one_hot(y, num_classes=self.hl_model.cfg.d_vocab_out)
    return y

  def get_correct_input(self, input):
    if isinstance(input, tuple) or isinstance(input, list):
      return input[0]
    elif isinstance(input, torch.Tensor):
      return input
    else:
      raise ValueError(f"Invalid input type: {type(input)}")

  def forward(self, input):
    x = self.get_correct_input(input)
    out = self.hl_model(x)
    return self.create_hl_output(out)

  def run_with_hooks(self, input, *args, **kwargs):
    x = self.get_correct_input(input)
    out = self.hl_model.run_with_hooks(x, *args, **kwargs)
    return self.create_hl_output(out)

  def run_with_cache(self, input):
    x = input[0]
    out, cache = self.hl_model.run_with_cache(x)
    return self.create_hl_output(out), cache

  def __call__(self, *args: Any, **kwds: Any) -> Any:
    return self.forward(*args, **kwds)
