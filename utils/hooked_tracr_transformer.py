from typing import List, Literal, Any, Union

import einops
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch as t
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer, HookedTransformerConfig

from tracr.compiler import assemble
from tracr.craft import vectorspace_fns
from tracr.craft.bases import BasisDirection, VectorSpaceWithBasis
from tracr.transformer.encoder import CategoricalEncoder

HookedTracrTransformerBatchInput = List[List[Any]]
HookedTracrTransformerReturnType = Literal["logits", "decoded"]


class HookedTracrTransformer(HookedTransformer):
  """A TransformerLens model built from a Tracr model."""

  def __init__(self,
               tracr_model: assemble.AssembledTransformerModel,
               device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
               *args, **kwargs) -> None:
    """Converts a tracr model to a transformer_lens model.
    Inspired by https://github.com/neelnanda-io/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb"""
    super().__init__(cfg=self.extract_tracr_config(tracr_model), *args, **kwargs)

    self.device = device
    self.tracr_input_encoder = tracr_model.input_encoder
    self.tracr_output_encoder = tracr_model.output_encoder
    self.residual_stream_labels = tracr_model.residual_labels

    sd = self.extract_tracr_state_dict(tracr_model)
    self.load_tracr_state_dict(sd)

    if "use_hook_mlp_in" in self.cfg.to_dict(): # Tracr models always include MLPs
        self.set_use_hook_mlp_in(True)

    self.freeze_all_weights()

  def __call__(self, *args, **kwargs):
    """Applies the internal transformer_lens model to an input."""
    if isinstance(args[0], list) or isinstance(args[0], pd.Series):
      # Input is a HookedTracrTransformerBatchInput
      return self.run_tracr_input(*args, **kwargs)
    else:
      # Input is a Tensor
      return super().__call__(*args, **kwargs)

  def run_tracr_input(self, batch_input: HookedTracrTransformerBatchInput,
                      return_type: HookedTracrTransformerReturnType = "logits") -> HookedTracrTransformerBatchInput | Float[Tensor, "batch_size seq_len d_vocab_out"]:
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
    bos_token = self.tracr_input_encoder.bos_token

    if self.is_categorical():
      logits = logits.argmax(dim=-1)
    else:
      logits = logits.squeeze(dim=-1)

    # The output has unspecified behavior for the BOS token, so we just add it back in after decoding.
    decoded_output_with_bos = [[bos_token] + self.tracr_output_encoder.decode(output)[1:] for output in logits.tolist()]

    return decoded_output_with_bos

  def load_tracr_state_dict(self, sd: dict[str, np.ndarray|jnp.ndarray]) -> None:
    """Creates a transformer_lens model from a config and state dict."""

    # Convert weights to tensors and load into the tl_model
    for k, v in sd.items():
      # Map Jax array to numpy array
      sd[k] = t.tensor(np.array(v)).to(self.device)

    self.load_state_dict(sd, strict=False)

  def extract_tracr_config(self, model: assemble.AssembledTransformerModel) -> HookedTransformerConfig:
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
    d_vocab_out = self.get_tracr_model_output_space(model).num_dims

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

  def extract_tracr_state_dict(self, model: assemble.AssembledTransformerModel) -> dict[str, np.ndarray | jnp.ndarray]:
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
        d_head = self.cfg.d_head,
        n_heads = self.cfg.n_heads
      )
      sd[f"blocks.{l}.attn.b_K"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/key"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = self.cfg.d_head,
        n_heads = self.cfg.n_heads
      )
      sd[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/query"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head = self.cfg.d_head,
        n_heads = self.cfg.n_heads
      )
      sd[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/query"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = self.cfg.d_head,
        n_heads = self.cfg.n_heads
      )
      sd[f"blocks.{l}.attn.W_V"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/value"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head = self.cfg.d_head,
        n_heads = self.cfg.n_heads
      )
      sd[f"blocks.{l}.attn.b_V"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/value"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = self.cfg.d_head,
        n_heads = self.cfg.n_heads
      )
      sd[f"blocks.{l}.attn.W_O"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/linear"]["w"],
        "(n_heads d_head) d_model -> n_heads d_head d_model",
        d_head = self.cfg.d_head,
        n_heads = self.cfg.n_heads
      )
      sd[f"blocks.{l}.attn.b_O"] = model.params[f"transformer/layer_{l}/attn/linear"]["b"]

      sd[f"blocks.{l}.mlp.W_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"]
      sd[f"blocks.{l}.mlp.b_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["b"]
      sd[f"blocks.{l}.mlp.W_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["w"]
      sd[f"blocks.{l}.mlp.b_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["b"]

    return sd

  def get_tracr_model_residual_space(self, model: assemble.AssembledTransformerModel) -> VectorSpaceWithBasis:
    residual_space = []
    for label in model.residual_labels:
      if ":" in label:
        # basis has name and value
        basis_name_and_value = label.split(":")
        residual_space.append(BasisDirection(basis_name_and_value[0], self.int_or_string(basis_name_and_value[1])))
      else:
        # basis has only name
        residual_space.append(BasisDirection(label, None))
    return VectorSpaceWithBasis(residual_space)

  def get_tracr_model_output_space(self, model: assemble.AssembledTransformerModel):
    return VectorSpaceWithBasis(model.output_encoder.basis)

  def is_categorical(self):
    """Returns true if the output_encoder is instance of CategoricalEncoder.
    False means that the output_encoder is instance of NumericalEncoder.
    """
    return isinstance(self.tracr_output_encoder, CategoricalEncoder)

  def freeze_all_weights(self):
    """Freezes all weights in the model."""
    for param in self.parameters():
      param.requires_grad = False

  def int_or_string(self, value):
    """Converts a value to an int if possible, otherwise returns the value as a string.
    THIS IS A HACK: The proper way to do this would be for model.params to contain an entry for "unembed".
    """
    try:
      return int(value)
    except ValueError:
      return value

  def to(self,
        device_or_dtype: Union[t.device, str, t.dtype],
        print_details: bool = True,):
    """Moves the model to a device and updates the device in the config."""
    self.device = device_or_dtype
    return super().to(device_or_dtype, print_details=print_details)