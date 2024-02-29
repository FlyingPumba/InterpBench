import os
from typing import Dict

import torch as t
import wandb
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.case_dataset import CaseDataset
from circuits_benchmark.training.compression.compressed_tracr_transformer_trainer import CompressedTracrTransformerTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput, HookedTracrTransformer


class NaturalCompressedTracrTransformerTrainer(CompressedTracrTransformerTrainer):
  def __init__(self,
               case: BenchmarkCase,
               original_model: HookedTracrTransformer,
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
    return f"case-{self.case.get_index()}-natural-resid-{self.compressed_model.cfg.d_model}"

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

  def compute_train_loss(self, batch: Dict[str, HookedTracrTransformerBatchInput]) -> Float[Tensor, ""]:
    # Run the input on both compressed and original model
    inputs = batch[CaseDataset.INPUT_FIELD]
    expected_outputs = batch[CaseDataset.CORRECT_OUTPUT_FIELD]
    predicted_outputs: Float[Tensor, ""] = self.get_compressed_model()(inputs)

    # Map expected outputs to tensor and drop BOS token from predictions
    if self.is_categorical:
      # remove BOS token from expected outputs
      expected_outputs = [e[1:] for e in expected_outputs]

      # use tracr original encoding to map expected_outputs. Also, convert all None values to NAN
      self.get_original_model().tracr_output_encoder.encoding_map[None] = t.nan
      encoded_expected_outputs = [self.get_original_model().tracr_output_encoder.encode(o) if o is not None else None
                                  for o in expected_outputs]
      expected_outputs = t.tensor(encoded_expected_outputs, device=self.device)

      # drop BOS token from predictions
      predicted_outputs = predicted_outputs[:, 1:]
    else:
      # Just drop the BOS token from the expected outputs
      expected_outputs = t.tensor([e[1:] for e in expected_outputs], device=self.device)

      # We drop the BOS token and squeeze because the predicted output has only one numerical element as output
      predicted_outputs = predicted_outputs[:, 1:].squeeze()

    # Set the same value to all positions in expected outputs that are NAN, and to the predicted outputs tensor
    # This will ensure that we are not comparing the loss of invalid positions (None values)
    mask_nan_values = t.isnan(expected_outputs)
    expected_outputs[mask_nan_values] = 0
    predicted_outputs[mask_nan_values] = 0

    # Calculate LOSS
    if self.is_categorical:
      # Make expected_outputs dtype back into int64
      expected_outputs = expected_outputs.long()

      # Cross entropy loss
      loss = -self.get_log_probs(predicted_outputs, expected_outputs).mean()
    else:
      # MSE loss
      loss = t.nn.functional.mse_loss(predicted_outputs, expected_outputs)

    if self.use_wandb:
      wandb.log({"train_loss": loss}, step=self.step)

    return loss

  def save_artifacts(self):
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    prefix = f"case-{self.case.get_index()}-resid-{self.compressed_model.cfg.d_model}"

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