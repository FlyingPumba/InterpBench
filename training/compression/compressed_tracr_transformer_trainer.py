from typing import List, Dict

import numpy as np
import torch as t
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.nn import Parameter
from transformer_lens import ActivationCache, HookedTransformer

from benchmark.benchmark_case import BenchmarkCase
from benchmark.case_dataset import CaseDataset
from training.compression.residual_stream_mapper.residual_stream_mapper import ResidualStreamMapper
from training.generic_trainer import GenericTrainer
from training.training_args import TrainingArgs
from utils.compare_tracr_output import compare_valid_positions, compare_positions_excluding_BOS
from utils.hooked_tracr_transformer import HookedTracrTransformerBatchInput
from utils.resampling_ablation_loss.resample_ablation_loss import get_resample_ablation_loss


class CompressedTracrTransformerTrainer(GenericTrainer):

  def __init__(self,
               case: BenchmarkCase,
               parameters: List[Parameter],
               training_args: TrainingArgs,
               is_categorical: bool,
               n_layers: int,
               output_dir: str | None = None):
    super().__init__(case, parameters, training_args, output_dir=output_dir)

    self.is_categorical = is_categorical
    self.n_layers = n_layers

  def setup_dataset(self):
    self.clean_dataset = self.case.get_clean_data(count=self.args.train_data_size)
    self.corrupted_dataset = self.case.get_corrupted_data(count=self.args.train_data_size)
    self.train_loader, self.test_loader = self.clean_dataset.train_test_split(self.args)

  def get_logits_and_cache_from_original_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    raise NotImplementedError

  def get_decoded_outputs_from_compressed_model(self, inputs: HookedTracrTransformerBatchInput) -> Tensor:
    raise NotImplementedError

  def get_logits_and_cache_from_compressed_model(
      self,
      inputs: HookedTracrTransformerBatchInput
  ) -> (Float[Tensor, "batch seq_len d_vocab"], ActivationCache):
    raise NotImplementedError

  def get_original_model(self) -> HookedTransformer:
    raise NotImplementedError

  def get_compressed_model(self) -> HookedTransformer:
    raise NotImplementedError

  def get_residual_stream_mapper(self) -> ResidualStreamMapper | None:
    return None

  def compute_train_loss(self, batch: Dict[str, HookedTracrTransformerBatchInput]) -> Float[Tensor, ""]:
    # Run the input on both compressed and original model
    inputs = batch[CaseDataset.INPUT_FIELD]
    compressed_model_logits, compressed_model_cache = self.get_logits_and_cache_from_compressed_model(inputs)
    original_model_logits, original_model_cache = self.get_logits_and_cache_from_original_model(inputs)

    if self.is_categorical:
      # Cross entropy loss
      loss = t.nn.functional.cross_entropy(compressed_model_logits.flatten(end_dim=-2),
                                           original_model_logits.flatten(end_dim=-2))
    else:
      # MSE loss
      loss = t.nn.functional.mse_loss(compressed_model_logits, original_model_logits)

    if self.use_wandb:
      wandb.log({"output_loss": loss}, step=self.step)

    # Sum the L2 of output vectors for all layers in both compressed and original model
    for layer in range(self.n_layers):
      compressed_model_output = compressed_model_cache["resid_post", layer]
      original_model_output = original_model_cache["resid_post", layer]

      layer_loss = t.nn.functional.mse_loss(compressed_model_output, original_model_output)
      if self.use_wandb:
        wandb.log({f"layer_{str(layer)}_loss": layer_loss}, step=self.step)

      loss += layer_loss

    if self.use_wandb:
      wandb.log({"train_loss": loss}, step=self.step)

    return loss

  def compute_test_metrics(self):
    test_data: Dict[str, HookedTracrTransformerBatchInput] = next(iter(self.test_loader))
    inputs = test_data[CaseDataset.INPUT_FIELD]
    expected_outputs = test_data[CaseDataset.CORRECT_OUTPUT_FIELD]
    predicted_outputs = self.get_decoded_outputs_from_compressed_model(inputs)

    correct_predictions = []
    expected_outputs_flattened = []
    predicted_outputs_flattened = []

    # The [1:] is for discarding the BOS token from comparison
    for predicted_output, expected_output in zip(predicted_outputs, expected_outputs):
      predictions = predicted_output[1:]
      expectations = expected_output[1:]

      if any(isinstance(p, str) for p in predictions):
        # We have chars, convert them to numbers using ord to avoid the torch issue: "too many dimensions 'str'".
        predictions = [ord(p) if p is not None else None for p in predictions]
        expectations = [ord(e) if e is not None else None for e in expectations]

      # Replace all predictions and expectations values where expectations have None with 0.
      # We do this so that we don't compare the loss of invalid positions (None values)
      indices = [i for i, e in enumerate(expectations) if e is None]
      for i in indices:
        predictions[i] = 0
        expectations[i] = 0

      predicted_outputs_flattened.extend(predictions)
      expected_outputs_flattened.extend(expectations)

      correct_predictions.extend(compare_positions_excluding_BOS(expectations,
                                                                 predictions,
                                                                 self.is_categorical,
                                                                 self.args.test_accuracy_atol))

    self.test_metrics["test_accuracy"] = np.mean(correct_predictions)

    predicted_outputs_tensor = t.tensor(predicted_outputs_flattened)
    expected_outputs_tensor = t.tensor(expected_outputs_flattened)

    if not self.is_categorical:
      self.test_metrics["test_mse"] = t.nn.functional.mse_loss(predicted_outputs_tensor,
                                                               expected_outputs_tensor).item()
    if self.args.resample_ablation_loss:
      # Compute the resampling ablation loss
      resample_ablation_loss_args = {
        "clean_inputs": self.case.get_clean_data(count=self.args.resample_ablation_data_size),
        "corrupted_inputs": self.case.get_corrupted_data(count=self.args.resample_ablation_data_size),
        "base_model": self.get_original_model(),
        "hypothesis_model": self.get_compressed_model(),
        "max_interventions": self.args.resample_ablation_max_interventions,
        "batch_size": self.args.resample_ablation_batch_size,
      }

      residual_stream_mapper = self.get_residual_stream_mapper()
      if residual_stream_mapper is not None:
        resample_ablation_loss_args["residual_stream_mapper"] = residual_stream_mapper

      self.test_metrics["resample_ablation_loss"] = get_resample_ablation_loss(
        **resample_ablation_loss_args
      ).item()

    if self.use_wandb:
      wandb.log(self.test_metrics, step=self.step)

  def define_wandb_metrics(self):
    super().define_wandb_metrics()
    wandb.define_metric("output_loss", summary="min")
    for layer in range(self.n_layers):
      wandb.define_metric(f"layer_{str(layer)}_loss", summary="min")
    if not self.is_categorical:
      wandb.define_metric("test_mse", summary="min")

  def get_wandb_config(self):
    cfg = super().get_wandb_config()
    return cfg.update({
      "is_categorical": self.is_categorical,
      "n_layers": self.n_layers,
      "original_resid_size": self.get_original_model().cfg.d_model,
      "compressed_resid_size": self.get_compressed_model().cfg.d_model,
    })
