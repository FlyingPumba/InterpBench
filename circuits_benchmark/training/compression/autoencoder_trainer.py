import torch as t
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.training.compression.autencoder import AutoEncoder
from circuits_benchmark.training.generic_trainer import GenericTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer


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
    _, tl_cache = self.tl_model.run_with_cache(tl_inputs)

    # collect the residual stream activations from all layers
    all_resid_pre = [tl_cache["resid_pre", layer] for layer in range(self.tl_model_n_layers)]
    last_resid_post = tl_cache["resid_post", self.tl_model_n_layers - 1]
    tl_activations = t.cat(all_resid_pre + [last_resid_post])

    # Shape of tl_activations is [activations_len, seq_len, d_model], we will convert to [activations_len, d_model] to
    # treat the residual stream for each sequence position as a separate sample.
    # The final length of tl_activations is train_data_size*(n_layers + 1)*seq_len
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

    batch_size = self.args.batch_size if self.args.batch_size is not None else len(train_data)
    self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

  def compute_train_loss(self, batch: Float[Tensor, "batch_size d_model"]) -> Float[Tensor, "batch posn-1"]:
    output = self.autoencoder(batch)

    loss = t.nn.functional.mse_loss(batch, output)

    if self.use_wandb:
      wandb.log({"train_loss": loss}, step=self.step)

    return loss

  def compute_test_metrics(self):
    # compute MSE and accuracy on each batch and take average at the end
    mse = t.tensor(0.0, device=self.device)
    accuracy = t.tensor(0.0, device=self.device)
    count = 0

    for inputs in self.test_loader:
      outputs = self.autoencoder(inputs)
      mse = mse + t.nn.functional.mse_loss(inputs, outputs)
      accuracy = accuracy + t.isclose(inputs, outputs, atol=self.args.test_accuracy_atol).float().mean()
      count = count + 1

    self.test_metrics["test_mse"] = (mse / count).item()
    self.test_metrics["test_accuracy"] = (accuracy / count).item()

    if self.use_wandb:
      wandb.log(self.test_metrics, step=self.step)

  def build_wandb_name(self):
    return f"case-{self.case.get_index()}-autoencoder-{self.autoencoder.compression_size}"

  def get_wandb_tags(self):
    tags = super().get_wandb_tags()
    tags.append("autoencoder-trainer")
    return tags

  def save_artifacts(self):
    prefix = f"case-{self.case.get_index()}-resid-{self.autoencoder.compression_size}"
    self.autoencoder.save(self.output_dir, prefix, self.wandb_run)