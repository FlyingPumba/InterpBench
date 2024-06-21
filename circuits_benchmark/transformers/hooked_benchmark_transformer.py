from __future__ import annotations

from typing import Callable, Optional, Tuple

from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import NamesFilter


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

  def get_caching_hooks(
      self,
      names_filter: NamesFilter = None,
      incl_bwd: bool = False,
      device=None,
      remove_batch_dim: bool = False,
      cache: Optional[dict] = None,
  ) -> Tuple[dict, list, list]:
    """Re-implementation of HookedTransformer.get_caching_hooks() that do not **detaches** the tensors by default.

    Creates hooks to cache activations. Note: It does not add the hooks to the model.

    Args:
        names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
        incl_bwd (bool, optional): Whether to also do backwards hooks. Defaults to False.
        device (_type_, optional): The device to store on. Keeps on the same device as the layer if None.
        remove_batch_dim (bool, optional): Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
        cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

    Returns:
        cache (dict): The cache where activations will be stored.
        fwd_hooks (list): The forward hooks.
        bwd_hooks (list): The backward hooks. Empty if incl_bwd is False.
    """
    if cache is None:
      cache = {}

    if names_filter is None:
      names_filter = lambda name: True
    elif type(names_filter) == str:
      filter_str = names_filter
      names_filter = lambda name: name == filter_str
    elif type(names_filter) == list:
      filter_list = names_filter
      names_filter = lambda name: name in filter_list
    self.is_caching = True

    def save_hook(tensor, hook):
      if remove_batch_dim:
        if self.weights_frozen:
          cache[hook.name] = tensor.detach().to(device)[0]
        else:
          cache[hook.name] = tensor.to(device)[0]
      else:
        if self.weights_frozen:
          cache[hook.name] = tensor.detach().to(device)
        else:
          cache[hook.name] = tensor.to(device)

    def save_hook_back(tensor, hook):
      if remove_batch_dim:
        if self.weights_frozen:
          cache[hook.name + "_grad"] = tensor.detach().to(device)[0]
        else:
          cache[hook.name + "_grad"] = tensor.to(device)[0]
      else:
        if self.weights_frozen:
          cache[hook.name + "_grad"] = tensor.detach().to(device)
        else:
          cache[hook.name + "_grad"] = tensor.to(device)

    fwd_hooks = []
    bwd_hooks = []
    for name, hp in self.hook_dict.items():
      if names_filter(name):
        fwd_hooks.append((name, save_hook))
        if incl_bwd:
          bwd_hooks.append((name, save_hook_back))

    return cache, fwd_hooks, bwd_hooks
