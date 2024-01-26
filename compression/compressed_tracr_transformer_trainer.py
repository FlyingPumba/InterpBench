from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import numpy as np
import torch as t
import wandb
from datasets import DatasetDict, Split, Dataset
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import ActivationCache

from benchmark.benchmark_case import BenchmarkCase
from compression.compressed_tracr_transformer import CompressedTracrTransformer
from utils.hooked_tracr_transformer import HookedTracrTransformerBatchInput


@dataclass
class CompressionTrainingArgs():
  batch_size: Optional[int] = 512
  epochs: Optional[int] = 1500
  lr: Optional[float] = 1e-3
  train_data_size: Optional[int] = 10000
  test_data_ratio: Optional[float] = 0.3
  wandb_project: Optional[str] = None
  wandb_name: Optional[str] = None
  test_accuracy_atol: Optional[float] = 1e-2


class CompressedTracrTransformerTrainer:
  def __init__(self, case: BenchmarkCase,
               model: CompressedTracrTransformer,
               args: CompressionTrainingArgs):
    super().__init__()
    self.case = case
    self.model = model
    self.device = model.device
    self.is_categorical = self.model.get_tl_model().is_categorical()
    self.n_layers = self.model.get_tl_model().cfg.n_layers

    self.args = args
    self.use_wandb = self.args.wandb_project is not None

    self.step = 0
    self.train_loss = np.nan
    self.test_metrics = {}
    self.dataset =  self.case.get_clean_data(count=args.train_data_size)
    self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.lr)

    self.split_dataset(args)

    if self.use_wandb and self.args.wandb_name is None:
      self.args.wandb_name = f"case-{self.case}-resid-{self.model.residual_stream_compression_size}"

  def split_dataset(self, args):
    """Split the dataset into train and test sets."""

    def custom_collate(items: List[Dict[str, List[Any]]]) -> dict[str, HookedTracrTransformerBatchInput]:
      return {BenchmarkCase.DATASET_INPUT_FIELD: [item[BenchmarkCase.DATASET_INPUT_FIELD] for item in items],
              BenchmarkCase.DATASET_CORRECT_OUTPUT_FIELD: [item[BenchmarkCase.DATASET_CORRECT_OUTPUT_FIELD] for item in
                                                           items]}

    split: DatasetDict = self.dataset.train_test_split(test_size=int(len(self.dataset) * args.test_data_ratio))

    self.train_loader = DataLoader(split[Split.TRAIN], batch_size=args.batch_size, shuffle=True,
                                   collate_fn=custom_collate)
    self.test_loader = DataLoader(split[Split.TEST], batch_size=len(split[Split.TEST]), shuffle=False,
                                  collate_fn=custom_collate)

  def train(self):
    """
    Trains the model, for `self.args.epochs` epochs.
    """
    print(f'Starting run to compress residual stream from {self.model.original_residual_stream_size} to'
          f' {self.model.residual_stream_compression_size} dimensions.')

    if self.use_wandb:
      wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)

    progress_bar = tqdm(total = len(self.train_loader) * self.args.epochs)

    for epoch in range(self.args.epochs):
      for i, batch in enumerate(self.train_loader):
        self.train_loss = self.training_step(batch)
        progress_bar.update()
        progress_bar.set_description(f"Epoch {epoch+1}, train_loss: {self.train_loss:.3f}" +
                                     self.build_test_metrics_string())

      self.evaluate_test_metrics()

    if self.use_wandb:
      wandb.finish()

  def training_step(self, batch: Dict[str, HookedTracrTransformerBatchInput]) -> Float[Tensor, ""]:
    '''
    Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

    Remember that `batch` is a dictionary with the single key 'tokens'.
    '''
    self.optimizer.zero_grad()

    # Run the input on both compressed and original model
    input = batch[BenchmarkCase.DATASET_INPUT_FIELD]
    compressed_model_logits, compressed_model_cache = self.model.run_with_cache(input)
    original_model_logits, original_model_cache = self.model.run_with_cache_on_original(input)

    # compute the loss
    loss = self.compute_loss(
      compressed_model_logits,
      compressed_model_cache,
      original_model_logits,
      original_model_cache
    )

    loss.backward()
    self.optimizer.step()

    self.step += 1

    return loss

  def compute_loss(
      self,
      compressed_model_logits: Float[Tensor, "batch seq_len d_vocab"],
      compressed_model_cache: ActivationCache,
      original_model_logits: Float[Tensor, "batch seq_len d_vocab"],
      original_model_cache: ActivationCache,
  ) -> Float[Tensor, "batch posn-1"]:
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

    return loss

  def evaluate_test_metrics(self):
    test_data = next(iter(self.test_loader))
    inputs = test_data[BenchmarkCase.DATASET_INPUT_FIELD]
    expected_outputs = test_data[BenchmarkCase.DATASET_CORRECT_OUTPUT_FIELD]
    predicted_outputs = self.model(inputs, return_type="decoded")

    correct_predictions = []
    expected_outputs_flattened = []
    predicted_outputs_flattened = []

    # The [1:] is for discarding the BOS token from comparison
    for predicted_output, expected_output in zip(predicted_outputs, expected_outputs):
      if self.model.get_tl_model().is_categorical():
        predictions = [ord(p) for p in predicted_output[1:]]
        expectations = [ord(e) for e in expected_output[1:]]
        predicted_outputs_flattened.extend(predictions)
        expected_outputs_flattened.extend(expectations)

        correct_predictions.extend(p == e for p, e in zip(predictions, expectations))
      else:
        predictions = [float(p) for p in predicted_output[1:]]
        expectations = [float(e) for e in expected_output[1:]]
        predicted_outputs_flattened.extend(predictions)
        expected_outputs_flattened.extend(expectations)

        close_predictions = np.isclose(predictions, expectations, atol=self.args.test_accuracy_atol)
        correct_predictions.extend(close_predictions.tolist())

    self.test_metrics["test_accuracy"] = np.mean(correct_predictions)

    predicted_outputs_tensor = t.tensor(predicted_outputs_flattened)
    expected_outputs_tensor = t.tensor(expected_outputs_flattened)

    if not self.model.get_tl_model().is_categorical():
      self.test_metrics["test_mse"] = t.nn.functional.mse_loss(predicted_outputs_tensor,
                                                               expected_outputs_tensor).item()


    if self.use_wandb:
      wandb.log(self.test_metrics, step=self.step)

  def build_test_metrics_string(self):
    if len(self.test_metrics.items()) == 0:
      return ""
    else:
      return ", " + ("".join([f"{k}: {v:.3f}, " for k, v in self.test_metrics.items()]))[:-2]