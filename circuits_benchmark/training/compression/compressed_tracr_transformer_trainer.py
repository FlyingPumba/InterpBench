import random
from functools import partial
from typing import List, Dict, Set, Optional

import numpy as np
import torch as t
import wandb
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import Parameter
from transformer_lens import ActivationCache, HookedTransformer

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.case_dataset import CaseDataset
from circuits_benchmark.metrics.iia import is_qkv_granularity_hook, regular_intervention_hook_fn
from circuits_benchmark.metrics.resampling_ablation_loss.resample_ablation_loss import \
  get_resample_ablation_loss_from_inputs
from circuits_benchmark.metrics.sparsity import get_zero_weights_pct
from circuits_benchmark.training.compression.activation_mapper.activation_mapper import ActivationMapper
from circuits_benchmark.training.generic_trainer import GenericTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.acdc_circuit_builder import get_full_acdc_circuit
from circuits_benchmark.transformers.circuit_node import CircuitNode
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

    if self.args.resample_ablation_test_loss:
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

  def get_activation_mapper(self) -> ActivationMapper | None:
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
      expected, predicted = replace_invalid_positions(expected_output, predicted_output, 0)

      if any(isinstance(p, str) for p in predicted):
        # We have chars, convert them to numbers to avoid the torch issue: "too many dimensions 'str'".
        predicted = [self.get_original_model().tracr_output_encoder.encoding_map[p] if isinstance(p, str) else p for p in predicted]
        expected = [self.get_original_model().tracr_output_encoder.encoding_map[e] if isinstance(e, str) else e for e in expected]

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

    # measure the effect of each node on the compressed model's output
    node_effect_results = self.evaluate_node_effect(
      self.case.get_clean_data(count=500, seed=random.randint(0, 1000000)),
      self.case.get_corrupted_data(count=500, seed=random.randint(0, 1000000)),
    )

    for node_str, result in node_effect_results.items():
      self.test_metrics[f"{node_str}_node_effect"] = result

    self.test_metrics["iia"] = self.sample_iia(self.clean_dataset, self.corrupted_dataset)

    if self.args.resample_ablation_test_loss:
      if self.epochs_since_last_test_resample_ablation_loss >= self.args.resample_ablation_loss_epochs_gap:
        self.epochs_since_last_test_resample_ablation_loss = 0

        # Compute the resampling ablation loss
        resample_ablation_loss_args = {
          "clean_inputs": self.case.get_clean_data(count=self.args.resample_ablation_data_size,
                                                   seed=random.randint(0, 1000000)),
          "corrupted_inputs": self.case.get_corrupted_data(count=self.args.resample_ablation_data_size,
                                                           seed=random.randint(0, 1000000)),
          "base_model": self.get_original_model(),
          "hypothesis_model": self.get_compressed_model(),
          "max_interventions": self.args.resample_ablation_max_interventions,
          "max_components": self.args.resample_ablation_max_components,
          "batch_size": self.args.resample_ablation_batch_size,
          "is_categorical": self.is_categorical,
        }

        activation_mapper = self.get_activation_mapper()
        if activation_mapper is not None:
          resample_ablation_loss_args["activation_mapper"] = activation_mapper

        resample_ablation_output = get_resample_ablation_loss_from_inputs(**resample_ablation_loss_args)
        self.test_metrics["test_resample_ablation_loss"] = resample_ablation_output.loss
        self.test_metrics["test_resample_ablation_var_exp"] = resample_ablation_output.variance_explained

        for hook_name, loss in resample_ablation_output.max_loss_per_hook.items():
          self.test_metrics[f"test_{hook_name}_max_cp_loss"] = loss.squeeze(-1)

        for hook_name, loss in resample_ablation_output.mean_loss_per_hook.items():
          self.test_metrics[f"test_{hook_name}_mean_cp_loss"] = loss

      self.epochs_since_last_test_resample_ablation_loss += 1

    # calculate sparsity metrics
    self.test_metrics["zero_weights_pct"] = get_zero_weights_pct(self.get_compressed_model())

    if self.use_wandb:
      wandb.log(self.test_metrics, step=self.step)

  def sample_iia(self, clean_data, corrupted_data, percentage_nodes_to_sample: Optional[float] = 0.2):
    iia = 0.0

    base_model = self.get_original_model()
    compressed_model = self.get_compressed_model()

    full_circuit = get_full_acdc_circuit(base_model.cfg.n_layers, base_model.cfg.n_heads)
    relevant_nodes: Set[CircuitNode] = set([node for node in full_circuit.nodes
                                       if "mlp_in" not in str(node) and not is_qkv_granularity_hook(str(node))])
    nodes_to_sample = random.sample(list(relevant_nodes), int(len(relevant_nodes) * percentage_nodes_to_sample))

    _, base_model_corrupted_cache = base_model.run_with_cache(corrupted_data.get_inputs())
    _, compressed_model_corrupted_cache = compressed_model.run_with_cache(corrupted_data.get_inputs())

    for node in nodes_to_sample:
      hook_name = node.name
      head_index = node.index

      # run clean data on both models, patching corrupted data where necessary
      base_model_patching_data = {}
      base_model_patching_data[hook_name] = base_model_corrupted_cache[hook_name]
      base_model_hook_fn = partial(regular_intervention_hook_fn, corrupted_cache=base_model_patching_data,
                                    head_index=head_index)

      compressed_model_patching_data = {}
      compressed_model_patching_data[hook_name] = compressed_model_corrupted_cache[hook_name]
      compressed_model_hook_fn = partial(regular_intervention_hook_fn, corrupted_cache=compressed_model_patching_data,
                                         head_index=head_index)

      with base_model.hooks([(hook_name, base_model_hook_fn)]):
        base_model_intervened_logits = base_model(clean_data.get_inputs())

      with compressed_model.hooks([(hook_name, compressed_model_hook_fn)]):
        compressed_model_intervened_logits = compressed_model(clean_data.get_inputs())

      # Remove BOS from logits
      base_model_intervened_logits = base_model_intervened_logits[:, 1:]
      compressed_model_intervened_logits = compressed_model_intervened_logits[:, 1:]

      if base_model.is_categorical():
        # apply log softmax to the logits
        base_model_intervened_logits: Float[Tensor, "batch pos vocab"] = t.nn.functional.log_softmax(base_model_intervened_logits, dim=-1)
        compressed_model_intervened_logits: Float[Tensor, "batch pos vocab"] = t.nn.functional.log_softmax(compressed_model_intervened_logits, dim=-1)

        # calculate labels for each position
        base_intervened_labels: Int[Tensor, "batch pos"] = t.argmax(base_model_intervened_logits, dim=-1)
        compressed_intervened_labels: Int[Tensor, "batch pos"] = t.argmax(compressed_model_intervened_logits, dim=-1)

        same_outputs_between_both_models_after_intervention = (
              base_intervened_labels == compressed_intervened_labels).all(dim=-1).float()
        iia = iia + same_outputs_between_both_models_after_intervention.mean().item()
      else:
        same_outputs_between_both_models_after_intervention = t.isclose(base_model_intervened_logits,
                                                                        compressed_model_intervened_logits,
                                                                        atol=self.args.test_accuracy_atol).float()
        iia = iia + same_outputs_between_both_models_after_intervention.mean().item()

    # return average over sampled nodes
    return iia / len(nodes_to_sample)

  def evaluate_node_effect(self, clean_data, corrupted_data):
    effect_by_node = {}

    full_circuit = get_full_acdc_circuit(self.get_original_model().cfg.n_layers, self.get_original_model().cfg.n_heads)
    all_nodes: Set[CircuitNode] = set(full_circuit.nodes)
    for node in all_nodes:
      hook_name = node.name
      head_index = node.index

      if "mlp_in" in hook_name:
        continue

      if is_qkv_granularity_hook(hook_name):
        continue

      compressed_model = self.get_compressed_model()
      compressed_model_original_logits = compressed_model(clean_data.get_inputs())
      _, compressed_model_corrupted_cache = compressed_model.run_with_cache(corrupted_data.get_inputs())

      compressed_model_patching_data = {}
      compressed_model_patching_data[hook_name] = compressed_model_corrupted_cache[hook_name]
      compressed_model_hook_fn = partial(regular_intervention_hook_fn, corrupted_cache=compressed_model_patching_data,
                                         head_index=head_index)
      with compressed_model.hooks([(hook_name, compressed_model_hook_fn)]):
        compressed_model_intervened_logits = compressed_model(clean_data.get_inputs())

      # Remove BOS from logits
      compressed_model_original_logits = compressed_model_original_logits[:, 1:]
      compressed_model_intervened_logits = compressed_model_intervened_logits[:, 1:]

      # apply log softmax to the logits
      compressed_model_original_logits: Float[Tensor, "batch pos vocab"] = t.nn.functional.log_softmax(
        compressed_model_original_logits, dim=-1)
      compressed_model_intervened_logits: Float[Tensor, "batch pos vocab"] = t.nn.functional.log_softmax(
        compressed_model_intervened_logits, dim=-1)

      # calculate labels for each position
      compressed_original_labels: Int[Tensor, "batch pos"] = t.argmax(compressed_model_original_logits, dim=-1)
      compressed_intervened_labels: Int[Tensor, "batch pos"] = t.argmax(compressed_model_intervened_logits, dim=-1)

      compressed_model_effect = (compressed_original_labels != compressed_intervened_labels).float().mean().item()
      effect_by_node[str(node)] = compressed_model_effect

    return effect_by_node

  def get_lr_validation_metric(self):
    metric = super().get_lr_validation_metric()
    if self.args.resample_ablation_test_loss:
      # our LR scheduler is maximizing, so we need to subtract the resample ablation loss from the metric
      metric = metric - self.test_metrics["test_resample_ablation_loss"]
    return metric

  def define_wandb_metrics(self):
    super().define_wandb_metrics()
    if not self.is_categorical:
      wandb.define_metric("test_mse", summary="min")
    if self.args.resample_ablation_test_loss:
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
