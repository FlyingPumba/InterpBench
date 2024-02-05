import os

import torch as t
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader

from benchmark.benchmark_case import BenchmarkCase
from training.compression.autencoder import AutoEncoder
from training.generic_trainer import GenericTrainer
from training.training_args import TrainingArgs
from utils.hooked_tracr_transformer import HookedTracrTransformer


class AutoEncoderTrainer(GenericTrainer):
  ACTIVATIONS_FIELD = "activations"

  def __init__(self,
               case: BenchmarkCase,
               autoencoder: AutoEncoder,
               tl_model: HookedTracrTransformer,
               args: TrainingArgs,
               output_dir: str | None = None):
    self.autoencoder = autoencoder
    self.tl_model = tl_model
    self.tl_model_n_layers = tl_model.cfg.n_layers
    self.tl_model.freeze_all_weights()
    super().__init__(case, list(autoencoder.parameters()), args, output_dir=output_dir)

  def setup_dataset(self):
    tl_dataset = self.case.get_clean_data(count=self.args.train_data_size)
    tl_inputs = tl_dataset.get_inputs()
    tl_output, tl_cache = self.tl_model.run_with_cache(tl_inputs)

    # collect the residual stream activations from all layers
    all_resid_pre = [tl_cache["resid_pre", layer] for layer in range(self.tl_model_n_layers)]
    last_resid_post = tl_cache["resid_post", self.tl_model_n_layers - 1]
    tl_activations = t.cat(all_resid_pre + [last_resid_post])

    # shape of tl_activations is [activations_len, seq_len, d_model], we will convert to [activations_len, d_model] to
    # treat the residual stream for each sequence position as a separate sample.
    tl_activations: Float[Tensor, "activations_len, seq_len, d_model"] = (
      tl_activations.transpose(0, 1).reshape(-1, tl_activations.shape[-1]))

    # shuffle tl_activations
    tl_activations = tl_activations[t.randperm(len(tl_activations))]

    if self.args.test_data_ratio is not None:
      # split the data into train and test sets
      test_data_size = int(len(tl_activations) * self.args.test_data_ratio)
      train_data_size = len(tl_activations) - test_data_size
      train_data, test_data = tl_activations.split([train_data_size, test_data_size])
    else:
      # use all data for training, and the same data for testing
      train_data = tl_activations
      test_data = tl_activations

    self.train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True)
    self.test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

  def compute_train_loss(self, inputs: Float[Tensor, "batch_size d_model"]) -> Float[Tensor, "batch posn-1"]:
    output = self.autoencoder(inputs)

    loss = t.nn.functional.mse_loss(inputs, output)

    if self.use_wandb:
      wandb.log({"train_loss": loss}, step=self.step)

    return loss

  def compute_test_metrics(self):
    inputs = next(iter(self.test_loader))
    outputs = self.autoencoder(inputs)

    self.test_metrics["test_mse"] = t.nn.functional.mse_loss(inputs, outputs).item()

    if self.use_wandb:
      wandb.log(self.test_metrics, step=self.step)

  def build_wandb_name(self):
    return f"case-{self.case.get_index()}-autoencoder-{self.autoencoder.compression_size}"

  def save_artifacts(self):
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    prefix = f"case-{self.case.get_index()}-resid-{self.autoencoder.compression_size}"

    # save the weights of the model using state_dict
    weights_path = os.path.join(self.output_dir, f"{prefix}-autoencoder-weights.pt")
    t.save(self.autoencoder.state_dict(), weights_path)

    # save the entire model
    model_path = os.path.join(self.output_dir, f"{prefix}-autoencoder.pt")
    t.save(self.autoencoder, model_path)

    if self.wandb_run is not None:
      # save the files as artifacts to wandb
      artifact = wandb.Artifact(f"{prefix}-autoencoder", type="model")
      artifact.add_file(weights_path)
      artifact.add_file(model_path)
      self.wandb_run.log_artifact(artifact)