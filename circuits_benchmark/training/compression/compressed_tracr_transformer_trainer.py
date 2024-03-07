from typing import List, Dict

import numpy as np
import torch as t
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.nn import Parameter
from transformer_lens import ActivationCache, HookedTransformer

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.case_dataset import CaseDataset
from circuits_benchmark.metrics.resampling_ablation_loss.resample_ablation_loss import get_resample_ablation_loss_from_inputs
from circuits_benchmark.metrics.sparsity import get_zero_weights_pct
from circuits_benchmark.training.compression.residual_stream_mapper.residual_stream_mapper import ResidualStreamMapper
from circuits_benchmark.training.generic_trainer import GenericTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformerBatchInput
from circuits_benchmark.utils.compare_tracr_output import replace_invalid_positions, compare_positions


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

    if self.args.resample_ablation_loss:
      self.epochs_since_last_test_resample_ablation_loss = self.args.resample_ablation_loss_epochs_gap

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

  def compute_test_metrics(self):
    test_data: Dict[str, HookedTracrTransformerBatchInput] = next(iter(self.test_loader))
    inputs = test_data[CaseDataset.INPUT_FIELD]
    expected_outputs = test_data[CaseDataset.CORRECT_OUTPUT_FIELD]
    predicted_outputs = self.get_decoded_outputs_from_compressed_model(inputs)

    correct_predictions = []
    expected_outputs_flattened = []
    predicted_outputs_flattened = []

    for predicted_output, expected_output in zip(predicted_outputs, expected_outputs):
      # Replace all predictions and expectations values where expectations have None, BOS, or PAD with 0.
      # We do this so that we don't compare the loss of invalid positions.
      expected, predicted = replace_invalid_positions(expected_output, predicted_output, 0.0)

      if any(isinstance(p, str) for p in predicted):
        # We have chars, convert them to numbers using ord to avoid the torch issue: "too many dimensions 'str'".
        predicted = [ord(p) if isinstance(p, str) else p for p in predicted]
        expected = [ord(e) if isinstance(e, str) else e for e in expected]

      predicted_outputs_flattened.extend(predicted)
      expected_outputs_flattened.extend(expected)

      correct_predictions.extend(compare_positions(expected,
                                                   predicted,
                                                   self.is_categorical,
                                                   self.args.test_accuracy_atol))

    self.test_metrics["test_accuracy"] = np.mean(correct_predictions)

    predicted_outputs_tensor = t.tensor(predicted_outputs_flattened)
    expected_outputs_tensor = t.tensor(expected_outputs_flattened)

    if not self.is_categorical:
      self.test_metrics["test_mse"] = t.nn.functional.mse_loss(predicted_outputs_tensor,
                                                               expected_outputs_tensor).item()

    if self.args.resample_ablation_loss:
      if self.epochs_since_last_test_resample_ablation_loss >= self.args.resample_ablation_loss_epochs_gap:
        self.epochs_since_last_test_resample_ablation_loss = 0

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

        resample_ablation_output = get_resample_ablation_loss_from_inputs(**resample_ablation_loss_args)
        self.test_metrics["test_resample_ablation_loss"] = resample_ablation_output.loss
        self.test_metrics["test_resample_ablation_var_exp"] = resample_ablation_output.variance_explained

      self.epochs_since_last_test_resample_ablation_loss += 1

    # calculate sparsity metrics
    self.test_metrics["zero_weights_pct"] = get_zero_weights_pct(self.get_compressed_model())

    if self.use_wandb:
      wandb.log(self.test_metrics, step=self.step)

  def get_lr_validation_metric(self):
    metric = super().get_lr_validation_metric()
    if self.args.resample_ablation_loss and "test_resample_ablation_loss" in self.test_metrics:
      # our LR scheduler is maximizing, so we need to subtract the resample ablation loss from the metric
      metric = metric - self.test_metrics["test_resample_ablation_loss"]
    return metric

  def define_wandb_metrics(self):
    super().define_wandb_metrics()
    if not self.is_categorical:
      wandb.define_metric("test_mse", summary="min")
    if self.args.resample_ablation_loss:
      wandb.define_metric("test_resample_ablation_loss", summary="min")
      wandb.define_metric("test_resample_ablation_var_exp", summary="max")

  def get_wandb_config(self):
    cfg = super().get_wandb_config()
    cfg.update({
      "is_categorical": self.is_categorical,
      "n_layers": self.n_layers,
      "original_resid_size": self.get_original_model().cfg.d_model,
      "compressed_resid_size": self.get_compressed_model().cfg.d_model,
    })
    return cfg
