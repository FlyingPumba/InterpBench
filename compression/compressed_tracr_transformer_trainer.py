from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import numpy as np
import torch
import torch as t
import wandb
from datasets import DatasetDict, Split, Dataset
from jaxtyping import Float, Bool
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
  def __init__(self, args: CompressionTrainingArgs,
               model: CompressedTracrTransformer,
               dataset: Dataset):
    super().__init__()
    self.model = model
    self.device = model.device
    self.is_categorical = self.model.get_tl_model().is_categorical()
    self.n_layers = self.model.get_tl_model().cfg.n_layers

    self.args = args
    self.use_wandb = self.args.wandb_project is not None

    self.step = 0
    self.dataset = dataset
    self.train_loss = np.nan
    self.test_loss = np.nan
    self.test_accuracy = np.nan
    self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.lr)

    self.split_dataset(args)

  def split_dataset(self, args):
    """Split the dataset into train and test sets."""

    def custom_collate(items: List[Dict[str, List[Any]]]) -> dict[str, HookedTracrTransformerBatchInput]:
      return {BenchmarkCase.DATASET_INPUT_FIELD: [item[BenchmarkCase.DATASET_INPUT_FIELD] for item in items],
              BenchmarkCase.DATASET_CORRECT_OUTPUT_FIELD: [item[BenchmarkCase.DATASET_CORRECT_OUTPUT_FIELD] for item in
                                                           items]}

    split: DatasetDict = self.dataset.train_test_split(test_size=int(len(self.dataset) * args.test_data_ratio))

    self.train_loader = DataLoader(split[Split.TRAIN], batch_size=args.batch_size, shuffle=True,
                                   collate_fn=custom_collate)
    self.test_loader = DataLoader(split[Split.TEST], batch_size=args.batch_size, shuffle=False,
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
        progress_bar.set_description(f"Epoch {epoch+1}, "
                                     f"train_loss: {self.train_loss:.3f}, "
                                     f"test_loss: {self.test_loss:.3f}, "
                                     f"test_accuracy: {self.test_accuracy:.2f}")

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
    batches_loss = []
    batches_accuracy = []

    for batch in self.test_loader:
      input = batch[BenchmarkCase.DATASET_INPUT_FIELD]
      expected_output = batch[BenchmarkCase.DATASET_CORRECT_OUTPUT_FIELD]

      # calculate test loss
      compressed_model_logits, compressed_model_cache = self.model.run_with_cache(input)
      original_model_logits, original_model_cache = self.model.run_with_cache_on_original(input)

      loss = self.compute_loss(
        compressed_model_logits,
        compressed_model_cache,
        original_model_logits,
        original_model_cache
      )
      batches_loss.append(loss.item())

      # calculate test accuracy
      predicted_output = self.model.tl_model.map_tl_output_to_tracr_output(compressed_model_logits)

      def compare_outputs(elem1: Any, elem2: Any):
        if self.model.get_tl_model().is_categorical():
          return elem1 == elem2
        else:
          return np.isclose(float(elem1), float(elem2), atol=self.args.test_accuracy_atol).item()

      # compare batched predicted vs expected output, element against element, discarding always the BOS token.
      correct_predictions = [compare_outputs(elem1, elem2)
                             for output1, output2 in zip(predicted_output, expected_output)
                             for elem1, elem2 in zip(output1[1:], output2[1:])]
      accuracy = np.mean([float(elem) for elem in correct_predictions])
      batches_accuracy.append(accuracy)

    self.test_loss = np.mean(batches_loss)
    self.test_accuracy = np.mean(batches_accuracy)