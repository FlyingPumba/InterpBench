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
  batch_size: Optional[int] = 16
  epochs: Optional[int] = 10
  lr: Optional[float] = 1e-3
  train_data_size: Optional[int] = 1000
  test_data_ratio: Optional[float] = 0.3
  wandb_project: Optional[str] = None
  wandb_name: Optional[str] = None


class CompressedTracrTransformerTrainer:
  def __init__(self, args: CompressionTrainingArgs,
               model: CompressedTracrTransformer,
               dataset: Dataset):
    super().__init__()
    self.model = model
    self.device = model.device
    self.args = args
    self.optimizer = t.optim.SGD(self.model.parameters(), lr=args.lr)
    self.use_wandb = self.args.wandb_project is not None
    self.step = 0
    self.dataset = dataset

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
      self.model.get_tl_model().is_categorical(),
      self.model.get_tl_model().cfg.n_layers,
      compressed_model_logits,
      compressed_model_cache,
      original_model_logits,
      original_model_cache
    )

    loss.backward()
    self.optimizer.step()

    self.step += 1

    if self.use_wandb:
      wandb.log({"train_loss": loss}, step=self.step)

    return loss

  def validation_step(self, batch: Dict[str, HookedTracrTransformerBatchInput]) -> Bool[Tensor, "bath_size x seq_len"]:
    '''
    Calculates & returns the accuracy on the tokens in the batch (i.e. how often the model's prediction
    is correct). Logging should happen in the `train` function (after we've computed the accuracy for
    the whole validation set).
    '''
    input = batch[BenchmarkCase.DATASET_INPUT_FIELD]
    expected_output = batch[BenchmarkCase.DATASET_CORRECT_OUTPUT_FIELD]
    predicted_output = self.model(input, return_type="decoded")

    def compare_outputs(elem1: Any, elem2: Any):
      if self.model.get_tl_model().is_categorical():
        return elem1 == elem2
      else:
        return np.isclose(float(elem1), float(elem2), atol=1.e-5).item()

    # compare batched predicted vs expected output, element against element, discarding always the BOS token.
    correct_predictions = [compare_outputs(elem1, elem2)
                           for output1, output2 in zip(predicted_output, expected_output)
                           for elem1, elem2 in zip(output1[1:], output2[1:])]
    return torch.tensor(correct_predictions).to(self.device)

  def train(self):
    '''
    Trains the model, for `self.args.epochs` epochs.
    '''
    print(f'Starting run to compress residual stream from {self.model.original_residual_stream_size} to'
          f' {self.model.residual_stream_compression_size} dimensions.')

    if self.use_wandb:
      wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)

    accuracy = np.nan

    batches_count = len(self.train_loader)
    progress_bar = tqdm(total = batches_count * self.args.epochs)

    for epoch in range(self.args.epochs):
      for i, batch in enumerate(self.train_loader):
        loss = self.training_step(batch)
        progress_bar.update()
        progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.2f}")

      correct_predictions = t.concat([self.validation_step(batch) for batch in self.test_loader])
      accuracy = correct_predictions.float().mean().item()
      if self.use_wandb:
        wandb.log({"accuracy": accuracy}, step=self.step)

    if self.use_wandb:
      wandb.finish()

  def compute_loss(
      self,
      is_categorical: bool,
      num_layers: int,
      compressed_model_logits: Float[Tensor, "batch seq_len d_vocab"],
      compressed_model_cache: ActivationCache,
      original_model_logits: Float[Tensor, "batch seq_len d_vocab"],
      original_model_cache: ActivationCache,
  ) -> Float[Tensor, "batch posn-1"]:
    if is_categorical:
      # Cross entropy loss
      loss = t.nn.functional.cross_entropy(compressed_model_logits.flatten(end_dim=-2),
                                           original_model_logits.flatten(end_dim=-2))
    else:
      # MSE loss
      loss = t.nn.functional.mse_loss(compressed_model_logits, original_model_logits)

    # Sum the L2 of output vectors for all layers in both compressed and original model
    for layer in range(num_layers):
      compressed_model_output = compressed_model_cache["resid_post", layer]
      original_model_output = original_model_cache["resid_post", layer]

      loss += t.nn.functional.mse_loss(compressed_model_output, original_model_output)

    return loss