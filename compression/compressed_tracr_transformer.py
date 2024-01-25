from typing import List, Tuple, Callable

import torch as t
from jaxtyping import Float
from torch import nn, Tensor
from transformer_lens.hook_points import HookPoint

from utils.hooked_tracr_transformer import HookedTracrTransformer, HookedTracrTransformerBatchInput, \
  HookedTracrTransformerReturnType


class CompressedTracrTransformer(nn.Module):

  def __init__(self,
               tl_model: HookedTracrTransformer,
               residual_stream_compression_size: int,
               device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")):
    super().__init__()
    self.residual_stream_compression_size = residual_stream_compression_size
    self.tl_model = tl_model
    self.init_range: float = 0.02
    self.num_layers = self.tl_model.cfg.n_layers
    self.device = device

    self.original_residual_stream_size = self.tl_model.cfg.d_model

    # To compress the model, we multiply with a matrix W when reading from
    # the residual stream, and with W^T when writing to the residual stream.

    # [to_size, from_size]
    self.W_compress = nn.Parameter(t.empty((self.residual_stream_compression_size,
                                            self.original_residual_stream_size), device=self.device))
    nn.init.normal_(self.W_compress, std=self.init_range)

    self.tl_model.reset_hooks(including_permanent=True)

  def get_tl_model(self) -> HookedTracrTransformer:
    return self.tl_model

  def build_hooks(self) -> List[Tuple[str, Callable]]:
    hooks = []

    write_to_compressed_residual = lambda x: x @ self.W_compress.T
    read_from_compressed_residual = lambda x: x @ self.W_compress

    def write_to_resid_hook_function(
        residual_stream: Float[Tensor, "batch seq_len d_model"],
        hook: HookPoint
    ) -> Float[Tensor, "batch seq_len d_model_compressed"]:
      return write_to_compressed_residual(residual_stream)

    def read_from_resid_hook_function(
        residual_stream: Float[Tensor, "batch seq_len n_head d_model_compressed"],
        hook: HookPoint
    ) -> Float[Tensor, "batch seq_len n_head d_model"]:
      return read_from_compressed_residual(residual_stream)

    # Add hooks for Attention heads and MLPs
    for hook_name in self.tl_model.hook_dict.keys():
      if "hook_k_input" in hook_name or "hook_q_input" in hook_name or "hook_v_input" in hook_name:
        # Attention head matrices read directly from the residual stream
        hooks.append((hook_name, read_from_resid_hook_function))
      elif "hook_attn_out" in hook_name:
        # the output of attention heads is written directly to the residual stream
        hooks.append((hook_name, write_to_resid_hook_function))
      elif "hook_mlp_out" in hook_name:
        # the output of the MLP is written directly to the residual stream
        hooks.append((hook_name, write_to_resid_hook_function))
      elif "hook_mlp_in" in hook_name:
        # the input of the MLP is read directly from the residual stream
        hooks.append((hook_name, read_from_resid_hook_function))

    # Add hooks for the pre-residual stream and the post-residual stream of all layers, so that the residual stream has
    # the same size as the original model.
    # Also, TransformerLens does not have a hook for the input of the unembedding, so we need to specially need to
    # change the dimension of the last residual stream back to whatever it was before compression.
    for layer in range(self.num_layers):
      hooks.append((f"blocks.{layer}.hook_resid_pre", write_to_resid_hook_function))
      hooks.append((f"blocks.{layer}.hook_resid_post", read_from_resid_hook_function))

    return hooks

  def __call__(self, tokens: HookedTracrTransformerBatchInput, return_type: HookedTracrTransformerReturnType="logits"):
    with self.tl_model.hooks(fwd_hooks=self.build_hooks()):
      return self.tl_model(tokens, return_type=return_type)

  def run_with_cache(self, tokens: HookedTracrTransformerBatchInput):
    with self.tl_model.hooks(fwd_hooks=self.build_hooks()):
      return self.tl_model.run_with_cache(tokens)

  def run_with_cache_on_original(self, tokens: HookedTracrTransformerBatchInput):
    return self.tl_model.run_with_cache(tokens)
