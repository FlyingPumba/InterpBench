from __future__ import annotations

from typing import Callable, Optional, Tuple

import einops
import torch as t
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCacheEntry
from transformer_lens.hook_points import NamesFilter


class HookedBenchmarkTransformer(HookedTransformer):
  """A small variation of the default implementation of HookedTransformer."""

  def __init__(self, remove_extra_tensor_cloning: bool = True, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.weights_frozen = False
    self.remove_extra_tensor_cloning = remove_extra_tensor_cloning

    if self.remove_extra_tensor_cloning:
      self.rewrite_forward_methods()

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

  def rewrite_forward_methods(self):
    """Rewrites the forward methods in all TransformerBlocks to avoid the unnecessary cloning of tensors."""
    for block in self.blocks:
      funcType = type(block.forward)
      # block.forward = funcType(transformer_block_forward_without_clones, block) # TODO: This causes acdc to fail on multiple heads. Either fix this or remove entirely.


def transformer_block_forward_without_clones(
    self,
    resid_pre: Float[t.Tensor, "batch pos d_model"],
    shortformer_pos_embed: Optional[
      Float[t.Tensor, "batch pos d_model"]
    ] = None,
    past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
    attention_mask: Optional[Int[t.Tensor, "batch offset_pos"]] = None,
) -> Float[t.Tensor, "batch pos d_model"]:
  """Same as the original TransformerBlock#forward method, but without the unnecessary cloning of tensors.

  A single Transformer block.

  Args:
      resid_pre (torch.Tensor): The residual stream - shape [batch, pos, d_model]
      cache (HookedTransformerKeyValueCache): A cache of previous keys and values, used only when generating text. Defaults to None.
      shortformer_pos_embed (torch.Tensor, optional): Only used for positional_embeddings_type == "shortformer". The positional embeddings. See HookedTransformerConfig for details. Defaults to None.
      attention_mask (torch.Tensor, optional): The attention mask for padded tokens. Defaults to None.

  Returns:
      _type_: _description_
  """
  resid_pre = self.hook_resid_pre(resid_pre)  # [batch, pos, d_model]

  def add_head_dimension(
      tensor: Float[t.Tensor, "batch pos d_model"],
      clone_tensor=True,
      # `einops.repeat` uses a view in torch, so we generally clone the tensor to avoid using shared storage for each head entry
  ):
    repeated_tensor = einops.repeat(
      tensor,
      "batch pos d_model -> batch pos n_heads d_model",
      n_heads=self.cfg.n_heads,
    )
    if clone_tensor:
      return repeated_tensor.clone()
    else:
      return repeated_tensor

  if self.cfg.use_attn_in or self.cfg.use_split_qkv_input:
    # We're adding a head dimension
    attn_in = add_head_dimension(resid_pre, clone_tensor=False)
    if shortformer_pos_embed is not None:
      shortformer_pos_embed = add_head_dimension(shortformer_pos_embed)
  else:
    attn_in = resid_pre

  if self.cfg.use_attn_in:
    attn_in = self.hook_attn_in(attn_in)

  if self.cfg.use_split_qkv_input:
    query_input = self.hook_q_input(attn_in)
    key_input = self.hook_k_input(attn_in)
    value_input = self.hook_v_input(attn_in)
  else:
    query_input = attn_in
    key_input = attn_in
    value_input = attn_in

  attn_out = self.hook_attn_out(
    # hook the residual stream states that are used to calculate the
    # queries, keys and values, independently.
    # Then take the layer norm of these inputs, and pass these to the attention module.
    self.attn(
      query_input=self.ln1(query_input)
                  + (0.0 if shortformer_pos_embed is None else shortformer_pos_embed),
      key_input=self.ln1(key_input)
                + (0.0 if shortformer_pos_embed is None else shortformer_pos_embed),
      value_input=self.ln1(value_input),
      past_kv_cache_entry=past_kv_cache_entry,
      attention_mask=attention_mask,
    )
  )  # [batch, pos, d_model]
  if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
    resid_mid = self.hook_resid_mid(
      resid_pre + attn_out
    )  # [batch, pos, d_model]
    mlp_in = (
      resid_mid
      if not self.cfg.use_hook_mlp_in
      else self.hook_mlp_in(resid_mid)
    )
    normalized_resid_mid = self.ln2(mlp_in)
    mlp_out = self.hook_mlp_out(
      self.mlp(normalized_resid_mid)
    )  # [batch, pos, d_model]
    resid_post = self.hook_resid_post(
      resid_mid + mlp_out
    )  # [batch, pos, d_model]
  elif self.cfg.parallel_attn_mlp:
    # Dumb thing done by GPT-J, both MLP and Attn read from resid_pre and write to resid_post, no resid_mid used.
    # In GPT-J, LN1 and LN2 are tied, in GPT-NeoX they aren't.
    normalized_resid_pre_2 = self.ln2(
      resid_pre
      if not self.cfg.use_hook_mlp_in
      else self.hook_mlp_in(resid_pre)
    )
    mlp_out = self.hook_mlp_out(
      self.mlp(normalized_resid_pre_2)
    )  # [batch, pos, d_model]
    resid_post = self.hook_resid_post(
      resid_pre + attn_out + mlp_out
    )  # [batch, pos, d_model]
  else:
    resid_post = self.hook_resid_post(
      resid_pre + attn_out
    )  # [batch, pos, d_model]
  return resid_post
