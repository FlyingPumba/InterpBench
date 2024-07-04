import random
from functools import partial
from typing import List, Set, Optional

import numpy as np
import torch as t
import wandb
from iit.utils.iit_dataset import train_test_split, IITDataset
from jaxtyping import Int
from torch import Tensor
from torch.nn import Parameter
from transformer_lens import HookedTransformer

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.metrics.iia import is_qkv_granularity_hook, regular_intervention_hook_fn
from circuits_benchmark.metrics.resampling_ablation_loss.resample_ablation_loss import \
  get_resample_ablation_loss
from circuits_benchmark.metrics.sparsity import get_zero_weights_pct
from circuits_benchmark.training.compression.activation_mapper.activation_mapper import ActivationMapper
from circuits_benchmark.training.generic_trainer import GenericTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.utils.circuit.circuit_eval import get_full_circuit
from circuits_benchmark.utils.circuit.circuit_node import CircuitNode


class CompressedTracrTransformerTrainer(GenericTrainer):

  def __init__(self,
               case: BenchmarkCase,
               parameters: List[Parameter],
               training_args: TrainingArgs,
               is_categorical: bool,
               n_layers: int,
               output_dir: str | None = None):
    self.is_categorical = is_categorical
    self.n_layers = n_layers
    self.effect_diffs_by_node = {}

    super().__init__(case, parameters, training_args, output_dir=output_dir)

    if self.args.resample_ablation_test_loss:
      self.epochs_since_last_test_resample_ablation_loss = self.args.resample_ablation_loss_epochs_gap

  def setup_dataset(self):
    dataset = self.case.get_clean_data(min_samples=20000, max_samples=120_000)
    train_dataset, test_dataset = train_test_split(
      dataset, test_size=self.args.test_data_ratio, random_state=42
    )
    self.train_dataset = IITDataset(train_dataset, train_dataset)
    self.test_dataset = IITDataset(test_dataset, test_dataset)

    self.train_loader = self.train_dataset.make_loader(batch_size=self.args.batch_size, num_workers=0)
    self.test_loader = self.test_dataset.make_loader(batch_size=self.args.batch_size, num_workers=0)

  def get_original_model(self) -> HookedTransformer:
    raise NotImplementedError

  def get_compressed_model(self) -> HookedTransformer:
    raise NotImplementedError

  def get_activation_mapper(self) -> ActivationMapper | None:
    return None

  def compute_test_metrics(self):
    clean_data = self.case.get_clean_data(max_samples=self.args.train_data_size, seed=random.randint(0, 1000000))

    inputs = clean_data.get_inputs()
    targets = clean_data.get_targets()
    predictions = self.get_compressed_model()(inputs)

    # drop BOS
    targets = targets[:, 1:]
    predictions = predictions[:, 1:]

    # calculate accuracy
    if self.is_categorical:
      # ues argmax on both predictions and targets to get the predicted and expected classes
      predicted_classes = t.argmax(predictions, dim=-1)
      expected_classes = t.argmax(targets, dim=-1)
      accuracy = (predicted_classes == expected_classes).float().mean().item()
    else:
      # use isclose to compare the predictions and targets
      accuracy = t.isclose(predictions, targets, atol=self.args.test_accuracy_atol).float().mean().item()

    self.test_metrics["test_accuracy"] = accuracy

    if not self.is_categorical:
      self.test_metrics["test_mse"] = t.nn.functional.mse_loss(predictions, targets).item()

    # measure the effect of each node on the compressed model's output
    compressed_model_node_effect_results = self.evaluate_node_effect(
      self.get_compressed_model(),
      self.test_dataset
    )

    for node_str, node_effect in compressed_model_node_effect_results.items():
      self.test_metrics[f"{node_str}_compressed_model_node_effect"] = node_effect

    self.test_metrics["avg_compressed_model_node_effect"] = np.mean(
      list(compressed_model_node_effect_results.values()))

    original_model_node_effect_results = self.evaluate_node_effect(
      self.get_original_model(),
      self.test_dataset
    )

    for node_str, node_effect in original_model_node_effect_results.items():
      self.test_metrics[f"{node_str}_original_model_node_effect"] = node_effect

    self.test_metrics["avg_original_model_node_effect"] = np.mean(list(original_model_node_effect_results.values()))

    # log abs diff of node effects
    avg_node_effect_diff = 0
    for node_str, original_model_node_effect in original_model_node_effect_results.items():
      compressed_model_node_effect = compressed_model_node_effect_results[node_str]
      node_effect_diff = abs(original_model_node_effect - compressed_model_node_effect)

      self.test_metrics[f"{node_str}_node_effect_diff"] = node_effect_diff
      self.effect_diffs_by_node[node_str] = node_effect_diff

      avg_node_effect_diff += node_effect_diff

    self.test_metrics["avg_node_effect_diff"] = avg_node_effect_diff / len(original_model_node_effect_results)

    dataset_loader = self.test_dataset.make_loader(batch_size=self.args.batch_size, num_workers=0)

    compressed_model = self.get_compressed_model()
    hl_ll_corr = self.case.get_correspondence(same_size=True)
    model_pair = self.case.build_model_pair(model_pair_name="strict",
                                            ll_model=compressed_model,
                                            hl_ll_corr=hl_ll_corr)
    eval_result = model_pair._run_eval_epoch(dataset_loader, model_pair.loss_fn)
    self.test_metrics["iia"] = eval_result.to_dict()["val/IIA"] / 100
    self.test_metrics["siia"] = eval_result.to_dict()["val/strict_accuracy"] / 100
    self.test_metrics["val/accuracy"] = eval_result.to_dict()["val/accuracy"] / 100
    self.test_metrics["val/iit_loss"] = eval_result.to_dict()["val/iit_loss"]

    if self.args.resample_ablation_test_loss:
      if self.epochs_since_last_test_resample_ablation_loss >= self.args.resample_ablation_loss_epochs_gap:
        self.epochs_since_last_test_resample_ablation_loss = 0

        # Compute the resampling ablation loss
        resample_ablation_loss_args = {
          "data": next(iter(self.test_loader)),
          "base_model": self.get_original_model(),
          "hypothesis_model": self.get_compressed_model(),
          "max_interventions": self.args.resample_ablation_max_interventions,
          "max_components": self.args.resample_ablation_max_components,
          "is_categorical": self.is_categorical,
        }

        activation_mapper = self.get_activation_mapper()
        if activation_mapper is not None:
          resample_ablation_loss_args["activation_mapper"] = activation_mapper

        resample_ablation_output = get_resample_ablation_loss(**resample_ablation_loss_args)
        self.test_metrics["test_resample_ablation_loss"] = resample_ablation_output.loss
        self.test_metrics["test_resample_ablation_var_exp"] = resample_ablation_output.variance_explained

        for hook_name, loss in resample_ablation_output.max_loss_per_node.items():
          self.test_metrics[f"test_{hook_name}_max_cp_loss"] = loss.squeeze(-1)

        for hook_name, loss in resample_ablation_output.mean_loss_per_node.items():
          self.test_metrics[f"test_{hook_name}_mean_cp_loss"] = loss

      self.epochs_since_last_test_resample_ablation_loss += 1

    # calculate sparsity metrics
    self.test_metrics["zero_weights_pct"] = get_zero_weights_pct(self.get_compressed_model())

    if self.use_wandb:
      wandb.log(self.test_metrics, step=self.step)

  def evaluate_node_effect(self, model, dataset: IITDataset):
    effect_by_node = {}

    full_circuit = get_full_circuit(self.get_original_model().cfg.n_layers, self.get_original_model().cfg.n_heads)
    all_nodes: Set[CircuitNode] = set(full_circuit.nodes)
    for node in all_nodes:
      hook_name = node.name
      head_index = node.index

      if "mlp_in" in hook_name:
        continue

      if is_qkv_granularity_hook(hook_name):
        continue

      clean_data, corrupted_data = next(iter(dataset.make_loader(batch_size=len(dataset), num_workers=0)))
      clean_inputs = clean_data[0]
      corrupted_inputs = corrupted_data[0]

      original_logits = model(clean_inputs)
      _, corrupted_cache = model.run_with_cache(corrupted_inputs)

      patching_data = {}
      patching_data[hook_name] = corrupted_cache[hook_name]
      hook_fn = partial(regular_intervention_hook_fn, corrupted_cache=patching_data,
                        head_index=head_index)
      with model.hooks([(hook_name, hook_fn)]):
        intervened_logits = model(clean_inputs)

      # Remove BOS from logits
      original_logits = original_logits[:, 1:]
      intervened_logits = intervened_logits[:, 1:]

      if self.case.is_categorical():
        # calculate labels for each position
        original_labels: Int[Tensor, "batch pos"] = t.argmax(original_logits, dim=-1)
        intervened_labels: Int[Tensor, "batch pos"] = t.argmax(intervened_logits, dim=-1)

        effect = (original_labels != intervened_labels).float().mean().item()
        effect_by_node[str(node)] = effect
      else:
        effect = 1 - t.isclose(original_logits, intervened_logits,
                               atol=self.args.test_accuracy_atol).float().mean().item()
        effect_by_node[str(node)] = effect

    return effect_by_node

  def get_lr_validation_metric(self):
    return self.test_metrics["siia"] + self.test_metrics["iia"] + super().get_lr_validation_metric()
  def check_early_stop_condition(self):
    return (self.args.early_stop_threshold is not None and
            self.test_metrics["test_accuracy"] >= self.args.early_stop_threshold and
            self.test_metrics["iia"] >= self.args.early_stop_threshold and
            self.test_metrics["siia"] >= self.args.early_stop_threshold)

  def build_test_metrics_string(self):
    return f", iia: {self.test_metrics.get('iia', 0):.3f}"

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
