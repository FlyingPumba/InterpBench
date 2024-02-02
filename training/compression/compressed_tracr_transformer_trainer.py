from typing import List

import numpy as np
import torch as t
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.nn import Parameter
from transformer_lens import ActivationCache, HookedTransformer

from benchmark.benchmark_case import BenchmarkCase
from benchmark.case_dataset import CaseDataset
from training.generic_trainer import GenericTrainer
from training.training_args import TrainingArgs
from utils.hooked_tracr_transformer import HookedTracrTransformerBatchInput
from utils.resampling_ablation_accuracy import get_resampling_ablation_accuracy


class CompressedTracrTransformerTrainer(GenericTrainer):

  def __init__(self,
               case: BenchmarkCase,
               parameters: List[Parameter],
               training_args: TrainingArgs,
               is_categorical: bool,
                n_layers: int):
    super().__init__(case, parameters, training_args)

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

  def compute_train_loss(self, batch: CaseDataset) -> Float[Tensor, ""]:
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
    test_data = next(iter(self.test_loader))
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

      if isinstance(predictions[0], str):
        # We have chars, convert them to numbers using ord to avoid the torch issue: "too many dimensions 'str'"
        predictions = [ord(p) for p in predictions]
        expectations = [ord(e) for e in expectations]

      predicted_outputs_flattened.extend(predictions)
      expected_outputs_flattened.extend(expectations)

      if self.is_categorical:
        correct_predictions.extend(p == e for p, e in zip(predictions, expectations))
      else:
        correct_predictions.extend(np.isclose(predictions, expectations, atol=self.args.test_accuracy_atol).tolist())

    self.test_metrics["test_accuracy"] = np.mean(correct_predictions)

    predicted_outputs_tensor = t.tensor(predicted_outputs_flattened)
    expected_outputs_tensor = t.tensor(expected_outputs_flattened)

    if not self.is_categorical:
      self.test_metrics["test_mse"] = t.nn.functional.mse_loss(predicted_outputs_tensor,
                                                               expected_outputs_tensor).item()
      self.test_metrics["resample_acc"] = get_resampling_ablation_accuracy(
          clean_inputs=inputs,
          corrupted_inputs=self.corrupted_dataset.get_inputs(),
          base_model=self.get_original_model(),
          hypothesis_model=self.get_compressed_model()
      ).item()

    if self.use_wandb:
      wandb.log(self.test_metrics, step=self.step)
