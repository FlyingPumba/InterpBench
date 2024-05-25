import torch as t
import wandb
from torch.utils.data import DataLoader
from transformer_lens import ActivationCache
from transformer_lens.components import MLP

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.case_dataset import CaseDataset
from circuits_benchmark.training.compression.autencoder import AutoEncoder
from circuits_benchmark.training.generic_trainer import GenericTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer


class MLPTrainer(GenericTrainer):

  def __init__(self,
               case: BenchmarkCase,
               mlp: MLP,
               original_model: HookedTracrTransformer,
               autoencoder: AutoEncoder,
               input_hook_name: str,
               output_hook_name: str,
               activations_cache: ActivationCache,
               args: TrainingArgs,
               output_dir: str | None = None):
    self.mlp = mlp
    self.original_model = original_model
    self.autoencoder = autoencoder
    self.autoencoder.freeze_all_weights()

    self.input_hook_name = input_hook_name
    self.output_hook_name = output_hook_name
    self.activations_cache = activations_cache

    super().__init__(case, list(mlp.parameters()), args, output_dir=output_dir)

  def setup_dataset(self):
    input_data = self.activations_cache[self.input_hook_name]
    output_data = self.activations_cache[self.output_hook_name]

    # compress the input and output data using the autoencoder
    with t.no_grad():
      input_data = self.autoencoder.encoder(input_data)
      output_data = self.autoencoder.encoder(output_data)

    # split the data into train and test sets
    assert self.args.test_data_ratio is not None, "test_data_ratio must be provided"

    # split the data into train and test sets
    data_size = input_data.shape[0]
    test_data_size = int(data_size * self.args.test_data_ratio)
    train_data_size = data_size - test_data_size

    train_input_data, test_input_data = t.split(input_data, [train_data_size, test_data_size])
    train_output_data, test_output_data = t.split(output_data, [train_data_size, test_data_size])

    # shuffle accordingly
    train_perm = t.randperm(train_data_size)
    test_perm = t.randperm(test_data_size)
    train_input_data = train_input_data[train_perm]
    train_output_data = train_output_data[train_perm]
    test_input_data = test_input_data[test_perm]
    test_output_data = test_output_data[test_perm]

    # prepare data loaders, using appropriate batch size
    self.batch_size = self.args.batch_size if self.args.batch_size is not None else train_data_size
    train_data = CaseDataset(train_input_data.tolist(), train_output_data.tolist())
    test_data = CaseDataset(test_input_data.tolist(), test_output_data.tolist())
    self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                   collate_fn=CaseDataset.custom_collate)
    self.test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False,
                                  collate_fn=CaseDataset.custom_collate)

  def compute_train_loss(self, batch):
    inputs = t.tensor(batch[CaseDataset.INPUT_FIELD])
    expected_outputs = t.tensor(batch[CaseDataset.CORRECT_OUTPUT_FIELD])

    mlp_output = self.mlp(inputs)
    loss = t.nn.functional.mse_loss(expected_outputs, mlp_output)

    if self.use_wandb:
      wandb.log({f"{self.output_hook_name}_train_loss": loss}, step=self.step)

    return loss

  def compute_test_metrics(self):
    # compute MSE and accuracy on each batch and take average at the end
    mse = t.tensor(0.0, device=self.device)
    accuracy = t.tensor(0.0, device=self.device)
    batches = 0

    for batch in self.test_loader:
      inputs = t.tensor(batch[CaseDataset.INPUT_FIELD])
      expected_outputs = t.tensor(batch[CaseDataset.CORRECT_OUTPUT_FIELD])

      mlp_output = self.mlp(inputs)

      mse = mse + t.nn.functional.mse_loss(expected_outputs, mlp_output)
      accuracy = accuracy + t.isclose(expected_outputs, mlp_output,
                                      atol=self.args.test_accuracy_atol).float().mean()

      batches = batches + 1

    mse = mse / batches
    self.test_metrics[f"{self.output_hook_name}_test_mse"] = mse.item()
    self.test_metrics[f"{self.output_hook_name}_test_accuracy"] = accuracy.item()

    if self.use_wandb:
      wandb.log(self.test_metrics, step=self.step)

  def get_lr_validation_metric(self):
    return self.test_metrics[f"{self.output_hook_name}_test_accuracy"]

  def get_early_stop_metric(self):
    return self.test_metrics[f"{self.output_hook_name}_test_accuracy"]

  def get_wandb_tags(self):
    tags = super().get_wandb_tags()
    tags.append("mlp-trainer")
    return tags

  def save_artifacts(self):
    prefix = f"case-{self.case.get_index()}-resid-{self.autoencoder.compression_size}"
    self.autoencoder.save(self.output_dir, prefix, self.wandb_run)