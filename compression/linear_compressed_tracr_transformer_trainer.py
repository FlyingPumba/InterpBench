
from typing import Dict

import numpy as np
import torch as t
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformer_lens import ActivationCache

from benchmark.benchmark_case import BenchmarkCase
from benchmark.case_dataset import CaseDataset
from compression.compression_training_args import CompressionTrainingArgs
from compression.linear_compressed_tracr_transformer import LinearCompressedTracrTransformer
from utils.hooked_tracr_transformer import HookedTracrTransformerBatchInput


class LinearCompressedTracrTransformerTrainer:
  def __init__(self, case: BenchmarkCase,
               model: LinearCompressedTracrTransformer,
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

    self.setup_dataset(args)

    # calculate number of epochs and steps
    assert self.args.epochs is not None or self.args.steps is not None, "Must specify either epochs or steps."
    self.epochs = self.args.epochs if self.args.epochs is not None else int(self.args.steps // len(self.train_loader)) + 1
    self.steps = self.args.steps if self.args.steps is not None else self.epochs * len(self.train_loader)

    self.optimizer = t.optim.AdamW(self.model.parameters(),
                                   lr=args.lr_start,
                                   weight_decay=args.weight_decay,
                                   betas=(args.beta_1, args.beta_2))

    # Learning rate scheduler with linear decay
    self.lr_warmup_steps = args.lr_warmup_steps
    if self.lr_warmup_steps is None:
      # by default, half of total steps
      self.lr_warmup_steps = self.steps // 2

    lr_lambda = lambda step: max(args.lr_end,
                                 args.lr_start - (args.lr_start - args.lr_end) * (step / self.lr_warmup_steps))
    self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

    if self.use_wandb and self.args.wandb_name is None:
      self.args.wandb_name = f"case-{self.case.index_str}-resid-{self.model.residual_stream_compression_size}"

  def setup_dataset(self, args):
    """Prepare the dataset and split it into train and test sets."""
    self.dataset = self.case.get_clean_data(count=args.train_data_size)
    self.train_loader, self.test_loader = self.dataset.train_test_split(args)

  def train(self):
    """
    Trains the model, for `self.args.epochs` epochs.
    """
    print(f'Starting run to compress residual stream from {self.model.original_residual_stream_size} to'
          f' {self.model.residual_stream_compression_size} dimensions.')

    if self.use_wandb:
      wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)

    progress_bar = tqdm(total=len(self.train_loader) * self.epochs)
    for epoch in range(self.epochs):
      for i, batch in enumerate(self.train_loader):
        self.train_loss = self.training_step(batch)

        progress_bar.update()
        progress_bar.set_description(f"Epoch {epoch + 1}, train_loss: {self.train_loss:.3f}" +
                                     self.build_test_metrics_string())

      self.evaluate_test_metrics()

      if (self.args.early_stop_test_accuracy is not None and
          self.test_metrics["test_accuracy"] >= self.args.early_stop_test_accuracy):
        break

    if self.use_wandb:
      wandb.finish()

    return {**self.test_metrics, "train_loss": self.train_loss.item()}

  def training_step(self, batch: Dict[str, HookedTracrTransformerBatchInput]) -> Float[Tensor, ""]:
    '''
    Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

    Remember that `batch` is a dictionary with the single key 'tokens'.
    '''
    self.optimizer.zero_grad()

    # Run the input on both compressed and original model
    input = batch[CaseDataset.INPUT_FIELD]
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
    self.lr_scheduler.step()

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

    if self.use_wandb:
      wandb.log({"train_loss": loss}, step=self.step)

    return loss

  def evaluate_test_metrics(self):
    test_data = next(iter(self.test_loader))
    inputs = test_data[CaseDataset.INPUT_FIELD]
    expected_outputs = test_data[CaseDataset.CORRECT_OUTPUT_FIELD]
    predicted_outputs = self.model(inputs, return_type="decoded")

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

      if self.model.get_tl_model().is_categorical():
        correct_predictions.extend(p == e for p, e in zip(predictions, expectations))
      else:
        correct_predictions.extend(np.isclose(predictions,
                                              expectations,
                                              atol=self.args.test_accuracy_atol).tolist())

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
