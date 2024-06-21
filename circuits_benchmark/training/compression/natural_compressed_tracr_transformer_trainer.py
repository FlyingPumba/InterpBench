import os
from typing import Dict, List, Any, Tuple

import numpy as np
import torch as t
import wandb
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookedRootModule

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.vocabs import TRACR_PAD, TRACR_BOS
from circuits_benchmark.training.compression.compressed_tracr_transformer_trainer import \
  CompressedTracrTransformerTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput, \
  HookedTracrTransformer
from circuits_benchmark.utils.compare_tracr_output import replace_invalid_positions_in_expected_outputs
from tracr.transformer.encoder import CategoricalEncoder


class NaturalCompressedTracrTransformerTrainer(CompressedTracrTransformerTrainer):
  def __init__(self,
               case: BenchmarkCase,
               original_model: HookedRootModule,
               compressed_model: HookedTracrTransformer,
               args: TrainingArgs,
               output_dir: str | None = None):
    self.original_model = original_model
    self.compressed_model = compressed_model
    super().__init__(case,
                     list(compressed_model.parameters()),
                     args,
                     original_model.is_categorical(),
                     original_model.cfg.n_layers,
                     output_dir=output_dir)

  def get_decoded_outputs_from_compressed_model(self, inputs: HookedTracrTransformerBatchInput) -> Tensor:
    return self.compressed_model(inputs, return_type="decoded")

  def get_logits_and_cache_from_compressed_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    return self.compressed_model.run_with_cache(inputs)

  def get_logits_and_cache_from_original_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    return self.original_model.run_with_cache(inputs)

  def get_original_model(self) -> HookedTracrTransformer:
    return self.original_model

  def get_compressed_model(self) -> HookedTransformer:
    return self.compressed_model

  def build_wandb_name(self):
    return f"case-{self.case.get_name()}-natural-resid-{self.compressed_model.cfg.d_model}"

  def get_wandb_tags(self):
    tags = super().get_wandb_tags()
    tags.append("natural-compression-trainer")
    return tags

  def get_log_probs(
      self,
      logits: Float[Tensor, "batch posn d_vocab"],
      tokens: Int[Tensor, "batch posn"]
  ) -> Float[Tensor, "batch posn-1"]:
    assert tokens.dtype == t.int64, "Tokens must be of type int64"

    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens

  def compute_train_loss(self, batch: Tuple[List[np.ndarray], List[np.ndarray]]) -> Float[Tensor, ""]:
    # Run the input on both compressed and original model
    inputs = batch[0]
    expected_outputs = batch[1]
    predicted_outputs: Float[Tensor, ""] = self.get_compressed_model()(inputs)

    def encode_output(output: List[Any]):
      encoder = self.get_original_model().tracr_output_encoder
      if isinstance(encoder, CategoricalEncoder):
        encoding = encoder.encoding_map
        return [encoding[x] if x not in [TRACR_BOS, TRACR_PAD, None] else x for x in output]
      else:
        return output

    # use tracr original encoding to map expected_outputs.
    expected_outputs = [encode_output(output) for output in expected_outputs]

    # Replace BOS, PAD, and None positions with 0
    expected_outputs, mask = replace_invalid_positions_in_expected_outputs(expected_outputs, predicted_outputs, -1)
    expected_outputs = t.tensor(expected_outputs, device=self.device)

    # retain only the elements in predicted_outputs and expected_outputs that have true in the mask
    predicted_outputs = predicted_outputs[~mask]
    expected_outputs = expected_outputs[~mask]

    # Calculate LOSS
    if self.is_categorical:
      # Make expected_outputs dtype back into int64
      expected_outputs = expected_outputs.long()

      # Cross entropy loss
      loss = -self.get_log_probs(predicted_outputs, expected_outputs).mean()
    else:
      # This benchmark uses numerical output. We squeeze to compare using MSE
      predicted_outputs = predicted_outputs.squeeze()
      loss = t.nn.functional.mse_loss(predicted_outputs, expected_outputs)

    if self.use_wandb:
      wandb.log({"train_loss": loss}, step=self.step)

    return loss

  def save_artifacts(self):
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    prefix = f"case-{self.case.get_name()}-resid-{self.compressed_model.cfg.d_model}"

    # save the weights of the model using state_dict
    weights_path = os.path.join(self.output_dir, f"{prefix}-natural-compression-weights.pt")
    t.save(self.compressed_model.state_dict(), weights_path)

    # save the entire model
    model_path = os.path.join(self.output_dir, f"{prefix}-natural-compressed-tracr-transformer.pt")
    t.save(self.compressed_model, model_path)

    if self.wandb_run is not None:
      # save the files as artifacts to wandb
      artifact = wandb.Artifact(f"{prefix}-naturally-compressed-tracr-transformer", type="model")
      artifact.add_file(weights_path)
      artifact.add_file(model_path)
      self.wandb_run.log_artifact(artifact)
