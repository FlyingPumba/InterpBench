from __future__ import annotations

from typing import List, Literal, Any, Union, Callable, Optional, Dict, Tuple

import einops
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch as t
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer, HookedTransformerConfig, HookedTransformerKeyValueCacheEntry
from transformer_lens.hook_points import NamesFilter

from tracr.compiler.assemble import AssembledTransformerModel
from tracr.craft import vectorspace_fns
from tracr.craft.bases import BasisDirection, VectorSpaceWithBasis
from tracr.transformer.encoder import CategoricalEncoder, Encoder

HookedTracrTransformerBatchInput = List[List[Any]]
HookedTracrTransformerReturnType = Literal["logits", "decoded"]


class HookedTracrTransformer(HookedTransformer):
  """A TransformerLens model built from a Tracr model."""

  def __init__(self,
               cfg: HookedTransformerConfig,
               tracr_input_encoder: Encoder,
               tracr_output_encoder: Encoder,
               residual_stream_labels: List[str],
               device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
               *args, **kwargs) -> None:
    """Converts a tracr model to a transformer_lens model.
    Inspired by https://github.com/neelnanda-io/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb"""
    super().__init__(cfg=cfg, *args, **kwargs)

    self.device = device
    self.tracr_input_encoder = tracr_input_encoder
    self.tracr_output_encoder = tracr_output_encoder
    self.residual_stream_labels = residual_stream_labels

    if "use_hook_mlp_in" in self.cfg.to_dict():  # Tracr models always include MLPs
      self.set_use_hook_mlp_in(True)

    self.weights_frozen = False
    self.rewrite_forward_methods()

  @classmethod
  def from_tracr_model(
      cls,
      tracr_model: AssembledTransformerModel,
      device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
  ) -> HookedTracrTransformer:
    """
    Initialize a HookedTracrTransformer from a Tracr model.
    """
    cfg = cls.extract_tracr_config(tracr_model)
    tl_model = cls(cfg, tracr_model.input_encoder, tracr_model.output_encoder, tracr_model.residual_labels, device)
    tl_model.load_weights_from_tracr_model(tracr_model)

    return tl_model

  @classmethod
  def from_hooked_tracr_transformer(cls,
                                    tl_model,
                                    overwrite_cfg_dict: Dict[str, Any] = None,
                                    init_params_fn: Optional[Callable[[Tensor], Tensor]] = None
                                    ) -> HookedTracrTransformer:
    """
    Initialize a HookedTracrTransformer from a HookedTracrTransformer.
    """
    cfg_dict = tl_model.cfg.to_dict()
    if overwrite_cfg_dict is not None:
      cfg_dict.update(overwrite_cfg_dict)
    cfg = HookedTransformerConfig.from_dict(cfg_dict)

    instance = cls(cfg,
                   tl_model.tracr_input_encoder,
                   tl_model.tracr_output_encoder,
                   tl_model.residual_stream_labels,
                   tl_model.device)

    if init_params_fn is not None:
      instance.reset_parameters(init_params_fn)
    else:
      instance.load_state_dict(tl_model.state_dict())

    return instance

  def __call__(self, *args, **kwargs):
    """Applies the internal transformer_lens model to an input."""
    if isinstance(args[0], list) or isinstance(args[0], pd.Series):
      # Input is a HookedTracrTransformerBatchInput
      return self.run_tracr_input(*args, **kwargs)
    else:
      # Input is a Tensor
      return super().__call__(*args, **kwargs)

  def run_tracr_input(self, batch_input: HookedTracrTransformerBatchInput,
                      return_type: HookedTracrTransformerReturnType = "logits") -> HookedTracrTransformerBatchInput | \
                                                                                   Float[
                                                                                     Tensor, "batch_size seq_len d_vocab_out"]:
    """Applies the internal transformer_lens model to an input."""
    tl_batch_input = self.map_tracr_input_to_tl_input(batch_input)
    logits = self(tl_batch_input)
    if return_type == "logits":
      return logits
    else:
      return self.map_tl_output_to_tracr_output(logits)

  def map_tracr_input_to_tl_input(self, batch_input: HookedTracrTransformerBatchInput) -> t.Tensor:
    """Maps a tracr input to a transformer_lens input."""
    encoding = [self.tracr_input_encoder.encode(input) for input in batch_input]
    return t.tensor(encoding).to(self.device)

  def map_tl_output_to_tracr_output(self, logits: t.Tensor) -> HookedTracrTransformerBatchInput:
    """Maps a transformer_lens output to a tracr output."""
    if self.is_categorical():
      logits = logits.argmax(dim=-1)
    else:
      logits = logits.squeeze(dim=-1)

    # The output has unspecified behavior for the BOS token, so we remove it and add it back in after decoding.
    bos_token = self.tracr_input_encoder.bos_token
    logits = logits[:, 1:]
    decoded_output_with_bos = [[bos_token] + self.tracr_output_encoder.decode(output)
                               for output in logits.tolist()]

    return decoded_output_with_bos

  def load_weights_from_tracr_model(self, tracr_model: AssembledTransformerModel) -> None:
    """Loads the weights from a tracr model into the transformer_lens model."""
    self.load_tracr_state_dict(self.extract_tracr_state_dict(tracr_model))

  def load_tracr_state_dict(self, sd: dict[str, np.ndarray | jnp.ndarray]) -> None:
    """Creates a transformer_lens model from a config and state dict."""

    # Convert weights to tensors and load into the tl_model
    for k, v in sd.items():
      # Map Jax array to numpy array
      sd[k] = t.tensor(np.array(v)).to(self.device)

    self.load_state_dict(sd, strict=False)

  @classmethod
  def extract_tracr_config(cls, model: AssembledTransformerModel) -> HookedTransformerConfig:
    """Extracts the configuration of a tracr model into a HookedTransformerConfig."""
    n_heads = model.model_config.num_heads
    n_layers = model.model_config.num_layers
    d_head = model.model_config.key_size
    d_mlp = model.model_config.mlp_hidden_size
    act_fn = "relu"
    normalization_type = "LN" if model.model_config.layer_norm else None
    attention_type = "causal" if model.model_config.causal else "bidirectional"

    #  Length of the input sequence
    n_ctx = model.params["pos_embed"]['embeddings'].shape[0]

    # Equivalent to length of vocab, with BOS and PAD at the end
    d_vocab = model.params["token_embed"]['embeddings'].shape[0]

    # Residual stream width
    d_model = model.params["token_embed"]['embeddings'].shape[1]

    # Number of dimensions in the residual stream used for the output
    d_vocab_out = cls.get_tracr_model_output_space(model).num_dims

    return HookedTransformerConfig(
      n_layers=n_layers,
      d_model=d_model,
      d_head=d_head,
      n_ctx=n_ctx,
      d_vocab=d_vocab,
      d_vocab_out=d_vocab_out,
      d_mlp=d_mlp,
      n_heads=n_heads,
      act_fn=act_fn,
      attention_dir=attention_type,
      normalization_type=normalization_type,
      use_attn_result=True,
      use_split_qkv_input=True,
      # device=device,
    )

  def extract_tracr_state_dict(self, model: AssembledTransformerModel) -> dict[str, np.ndarray | jnp.ndarray]:
    """Extracts the state dict of a tracr model into a dict."""
    sd = {}
    sd["pos_embed.W_pos"] = model.params["pos_embed"]['embeddings']
    sd["embed.W_E"] = model.params["token_embed"]['embeddings']

    # fetch output space and project residual space onto it to get the unembed matrix
    residual_space = self.get_tracr_model_residual_space(model)
    output_space = self.get_tracr_model_output_space(model)
    sd["unembed.W_U"] = vectorspace_fns.project(residual_space, output_space).matrix

    for l in range(self.cfg.n_layers):
      sd[f"blocks.{l}.attn.W_K"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/key"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head=self.cfg.d_head,
        n_heads=self.cfg.n_heads
      )
      sd[f"blocks.{l}.attn.b_K"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/key"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head=self.cfg.d_head,
        n_heads=self.cfg.n_heads
      )
      sd[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/query"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head=self.cfg.d_head,
        n_heads=self.cfg.n_heads
      )
      sd[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/query"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head=self.cfg.d_head,
        n_heads=self.cfg.n_heads
      )
      sd[f"blocks.{l}.attn.W_V"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/value"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head=self.cfg.d_head,
        n_heads=self.cfg.n_heads
      )
      sd[f"blocks.{l}.attn.b_V"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/value"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head=self.cfg.d_head,
        n_heads=self.cfg.n_heads
      )
      sd[f"blocks.{l}.attn.W_O"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/linear"]["w"],
        "(n_heads d_head) d_model -> n_heads d_head d_model",
        d_head=self.cfg.d_head,
        n_heads=self.cfg.n_heads
      )
      sd[f"blocks.{l}.attn.b_O"] = model.params[f"transformer/layer_{l}/attn/linear"]["b"]

      sd[f"blocks.{l}.mlp.W_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"]
      sd[f"blocks.{l}.mlp.b_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["b"]
      sd[f"blocks.{l}.mlp.W_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["w"]
      sd[f"blocks.{l}.mlp.b_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["b"]

    return sd

  def get_tracr_model_residual_space(self, model: AssembledTransformerModel) -> VectorSpaceWithBasis:
    residual_space = []
    for label in model.residual_labels:
      if ":" in label:
        # basis has name and value
        basis_name_and_value = label.split(":")
        residual_space.append(
          BasisDirection(basis_name_and_value[0], self.str_to_correct_type(basis_name_and_value[1])))
      else:
        # basis has only name
        residual_space.append(BasisDirection(label, None))
    return VectorSpaceWithBasis(residual_space)

  @staticmethod
  def get_tracr_model_output_space(model: AssembledTransformerModel):
    assert model.output_encoder is not None, "Tracr model must have an output encoder."
    return VectorSpaceWithBasis(model.output_encoder.basis)

  def is_categorical(self):
    """Returns true if the output_encoder is instance of CategoricalEncoder.
    False means that the output_encoder is instance of NumericalEncoder.
    """
    return isinstance(self.tracr_output_encoder, CategoricalEncoder)

  def freeze_all_weights(self):
    """Freezes all weights in the model."""
    self.weights_frozen = True
    for param in self.parameters():
      param.requires_grad = False

  def unfreeze_all_weights(self):
    """Unfreezes all weights in the autoencoder."""
    self.weights_frozen = False
    for param in self.parameters():
      param.requires_grad = True

  def reset_parameters(self, init_fn: Callable[[Tensor], Tensor]):
    """Resets all parameters in the model."""
    for name, param in self.named_parameters():
      init_fn(param)

  def str_to_correct_type(self, value):
    """Converts a value to the correct type, otherwise returns the value as a string.
    THIS IS A HACK: The proper way to do this would be for model.params to contain an entry for "unembed".
    """
    # try converting to int or bool
    if value == "True":
      return True

    if value == "False":
      return False

    try:
      return int(value)
    except ValueError:
      pass

    try:
      return float(value)
    except ValueError:
      pass

    # just return the string value
    return value

  def to(self,
         device_or_dtype: Union[t.device, str, t.dtype],
         print_details: bool = True):
    """Moves the model to a device and updates the device in the config."""
    self.device = device_or_dtype
    return super().to(device_or_dtype, print_details=print_details)

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
      block.forward = funcType(transformer_block_forward_without_clones, block)


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
