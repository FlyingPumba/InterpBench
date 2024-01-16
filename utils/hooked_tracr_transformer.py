import einops
import jax.numpy as jnp
import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tracr.compiler import assemble
from tracr.craft import vectorspace_fns
from tracr.craft.bases import BasisDirection, VectorSpaceWithBasis


class HookedTracrTransformer():

  def __init__(self, tracr_model: assemble.AssembledTransformerModel) -> None:
    """Converts a tracr model to a transformer_lens model.
    Inspired by https://github.com/neelnanda-io/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb"""
    self.input_encoder = tracr_model.input_encoder
    self.output_encoder = tracr_model.output_encoder

    cfg = self.extract_config(tracr_model)
    sd = self.extract_state_dict(tracr_model, cfg)
    self.tl_model = self.create_hooked_transformer(cfg, sd)
    self.cfg = cfg
    self.sd = sd

  def __call__(self, input: list[int]) -> list[int]:
    """Applies the internal transformer_lens model to an input."""
    tl_input = self.map_tracr_input_to_tl_input(input)
    logits = self.tl_model(tl_input)
    output = self.map_tl_output_to_tracr_output(logits)
    return output

  def map_tracr_input_to_tl_input(self, input: list[int]) -> torch.Tensor:
    """Maps a tracr input to a transformer_lens input."""
    encoding = self.input_encoder.encode(input)
    return torch.tensor(encoding).unsqueeze(dim=0)

  def map_tl_output_to_tracr_output(self, logits: torch.Tensor) -> list[int]:
    """Maps a transformer_lens output to a tracr output."""
    bos_token = self.input_encoder.bos_token

    max_output_indices = logits.squeeze(dim=0).argmax(dim=-1)
    decoded_output = self.output_encoder.decode(max_output_indices.tolist())

    # The outputhas have unspecified behavior for the BOS token, so we just add it back in
    decoded_output_with_bos = [bos_token] + decoded_output[1:]

    return decoded_output_with_bos

  def create_hooked_transformer(self, cfg: HookedTransformerConfig, sd: dict[str, np.ndarray|jnp.ndarray]) -> HookedTransformer:
    """Creates a transformer_lens model from a config and state dict."""
    # Create a blank HookedTransformer model
    tl_model = HookedTransformer(cfg)

    # Convert weights to tensors and load into the tl_model
    for k, v in sd.items():
      # Map Jax array to numpy array
      sd[k] = torch.tensor(np.array(v))

    tl_model.load_state_dict(sd, strict=False)

    return tl_model

  def extract_config(self, model: assemble.AssembledTransformerModel) -> HookedTransformerConfig:
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

    # Equivalent to length of vocab, WITHOUT BOS and PAD at the end because we never care about these outputs
    # In practice, we always feed the logits into an argmax
    d_vocab_out = model.params["token_embed"]['embeddings'].shape[0] - 2

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
    )

  def extract_state_dict(self, model: assemble.AssembledTransformerModel, cfg: HookedTransformerConfig) -> dict[str, np.ndarray|jnp.ndarray]:
    """Extracts the state dict of a tracr model into a dict."""
    sd = {}
    sd["pos_embed.W_pos"] = model.params["pos_embed"]['embeddings']
    sd["embed.W_E"] = model.params["token_embed"]['embeddings']
    # Equivalent to max_seq_len plus one, for the BOS

    # The unembed is just a projection onto the first few elements of the residual stream, these store output tokens
    # This is a NumPy array, the rest are Jax Arrays, but w/e it's fine.
    # sd["unembed.W_U"] = np.eye(cfg.d_model, cfg.d_vocab_out)

    # build residual space
    residual_space = []
    for label in model.residual_labels:
      if ":" in label:
        # basis has name and value
        basis_name_and_value = label.split(":")
        residual_space.append(BasisDirection(basis_name_and_value[0], self.int_or_string(basis_name_and_value[1])))
      else:
        # basis has only name
        residual_space.append(BasisDirection(label, None))
    residual_space = VectorSpaceWithBasis(residual_space)

    # fetch output space and project residual space onto it to get the unembed matrix
    output_space = VectorSpaceWithBasis(model.output_encoder.basis)
    sd["unembed.W_U"] = vectorspace_fns.project(residual_space, output_space).matrix

    for l in range(cfg.n_layers):
      sd[f"blocks.{l}.attn.W_K"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/key"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head = cfg.d_head,
        n_heads = cfg.n_heads
      )
      sd[f"blocks.{l}.attn.b_K"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/key"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = cfg.d_head,
        n_heads = cfg.n_heads
      )
      sd[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/query"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head = cfg.d_head,
        n_heads = cfg.n_heads
      )
      sd[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/query"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = cfg.d_head,
        n_heads = cfg.n_heads
      )
      sd[f"blocks.{l}.attn.W_V"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/value"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head = cfg.d_head,
        n_heads = cfg.n_heads
      )
      sd[f"blocks.{l}.attn.b_V"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/value"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = cfg.d_head,
        n_heads = cfg.n_heads
      )
      sd[f"blocks.{l}.attn.W_O"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/linear"]["w"],
        "(n_heads d_head) d_model -> n_heads d_head d_model",
        d_head = cfg.d_head,
        n_heads = cfg.n_heads
      )
      sd[f"blocks.{l}.attn.b_O"] = model.params[f"transformer/layer_{l}/attn/linear"]["b"]

      sd[f"blocks.{l}.mlp.W_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"]
      sd[f"blocks.{l}.mlp.b_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["b"]
      sd[f"blocks.{l}.mlp.W_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["w"]
      sd[f"blocks.{l}.mlp.b_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["b"]

    return sd

  def int_or_string(self, value):
    """Converts a value to an int if possible, otherwise returns the value as a string.
    THIS IS A HACK: The proper way to do this would be for model.params to contain an entry for "unembed".
    """
    try:
      return int(value)
    except ValueError:
      return value