import numpy as np
import torch as t
import wandb
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer

from benchmark.benchmark_case import BenchmarkCase
from benchmark.case_dataset import CaseDataset
from training.compression.compressed_tracr_transformer_trainer import CompressedTracrTransformerTrainer
from training.training_args import TrainingArgs
from utils.hooked_tracr_transformer import HookedTracrTransformerBatchInput, HookedTracrTransformer


class NaturalCompressedTracrTransformerTrainer(CompressedTracrTransformerTrainer):
  def __init__(self,
               case: BenchmarkCase,
               original_model: HookedTracrTransformer,
               compressed_model: HookedTracrTransformer,
               args: TrainingArgs):
    self.original_model = original_model
    self.compressed_model = compressed_model
    super().__init__(case,
                     list(compressed_model.parameters()),
                     args,
                     original_model.is_categorical(),
                     original_model.cfg.n_layers)

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

  def get_log_probs(
      self,
      logits: Float[Tensor, "batch posn d_vocab"],
      tokens: Int[Tensor, "batch posn"]
  ) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens

  def compute_train_loss(self, batch: CaseDataset) -> Float[Tensor, ""]:
    # Run the input on both compressed and original model
    inputs = batch[CaseDataset.INPUT_FIELD]
    expected_outputs = batch[CaseDataset.CORRECT_OUTPUT_FIELD]
    predicted_outputs: Float[Tensor, ""] = self.get_compressed_model()(inputs)

    if self.is_categorical:
      # remove BOS token from expected outputs
      expected_outputs = [e[1:] for e in expected_outputs]

      # use tracr original encoding to map expected_outputs
      expected_outputs = t.tensor([self.get_original_model().tracr_output_encoder.encode(o) for o in expected_outputs],
                                  device=self.device)

      # vocab size is 28, but logits size is 26

      # drop BOS token from predictions
      predicted_outputs = predicted_outputs[:, 1:]

      # Cross entropy loss
      loss = -self.get_log_probs(predicted_outputs, expected_outputs).mean()
    else:
      # Just drop the BOS token from the expected outputs
      expected_outputs = t.tensor([e[1:] for e in expected_outputs],
                                  device=self.device)

      # We drop the BOS token and squeeze because the predicted output has only one numerical element as output
      predicted_outputs = predicted_outputs[:, 1:].squeeze()

      # MSE loss
      loss = t.nn.functional.mse_loss(predicted_outputs, expected_outputs)

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

    if self.use_wandb:
      wandb.log(self.test_metrics, step=self.step)
