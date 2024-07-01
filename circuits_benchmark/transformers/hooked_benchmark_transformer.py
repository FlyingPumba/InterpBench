from __future__ import annotations

from typing import Callable

from torch import Tensor
from transformer_lens import HookedTransformer


class HookedBenchmarkTransformer(HookedTransformer):
  """A small variation of the default implementation of HookedTransformer."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.weights_frozen = False

  def freeze_all_weights(self):
    """Freezes all weights in the model."""
    self.weights_frozen = True
    for param in self.parameters():
      param.requires_grad = False

  def unfreeze_all_weights(self):
    """Unfreezes all weights in the model."""
    self.weights_frozen = False
    for param in self.parameters():
      param.requires_grad = True

  def reset_parameters(self, init_fn: Callable[[Tensor], Tensor]):
    """Resets all parameters in the model."""
    for name, param in self.named_parameters():
      init_fn(param)
