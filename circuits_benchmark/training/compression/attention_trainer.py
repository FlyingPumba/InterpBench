import einops
import torch as t
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import ActivationCache
from transformer_lens.components import MLP, Attention

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.benchmark.case_dataset import CaseDataset
from circuits_benchmark.training.compression.autencoder import AutoEncoder
from circuits_benchmark.training.generic_trainer import GenericTrainer
from circuits_benchmark.training.training_args import TrainingArgs
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer


class AttentionTrainer(GenericTrainer):

  def __init__(self,
               case: BenchmarkCase,
               attn: Attention,
               original_model: HookedTracrTransformer,
               autoencoder: AutoEncoder,
               input_hook_name: str,
               output_hook_name: str,
               activations_cache: ActivationCache,
               args: TrainingArgs,
               head_index: int | None = None,
               output_dir: str | None = None):
    self.attn = attn
    self.original_model = original_model
    self.autoencoder = autoencoder
    self.autoencoder.freeze_all_weights()

    self.input_hook_name = input_hook_name
    self.output_hook_name = output_hook_name
    self.head_index = head_index
    self.activations_cache = activations_cache

    super().__init__(case, list(attn.parameters()), args, output_dir=output_dir)

  def setup_dataset(self):
    input_data = self.activations_cache[self.input_hook_name]
    output_data = self.activations_cache[self.output_hook_name]

    if self.head_index is not None:
      output_data = output_data[:, :, self.head_index]

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

  def add_head_dimension(
      self,
      tensor: Float[Tensor, "batch pos d_model"],
      clone_tensor=True,
  ):
    repeated_tensor = einops.repeat(
      tensor,
      "batch pos d_model -> batch pos n_heads d_model",
      n_heads=self.original_model.cfg.n_heads,
    )
    return repeated_tensor.clone()

  def compute_train_loss(self, batch):
    inputs = t.tensor(batch[CaseDataset.INPUT_FIELD])
    expected_outputs = t.tensor(batch[CaseDataset.CORRECT_OUTPUT_FIELD])

    attn_output = self.run_attn(inputs)
    loss = t.nn.functional.mse_loss(expected_outputs, attn_output)

    if self.use_wandb:
      wandb.log({f"{self.output_hook_name}_train_loss": loss}, step=self.step)

    return loss

  def run_attn(self, inputs):
    attn_output = None

    def hook_fn(activation, hook=None):
      nonlocal attn_output
      attn_output = activation
      return activation

    # run the attention module with a hook to capture the intermediate output of heads before they are summed together.
    self.attn.hook_result.add_hook(hook_fn)
    self.attn(self.add_head_dimension(inputs),
              self.add_head_dimension(inputs),
              self.add_head_dimension(inputs))
    self.attn.hook_result.remove_hooks()

    if self.head_index is not None:
      attn_output = attn_output[:, :, self.head_index]

    # add output bias (not captured by the hook_result)
    attn_output = attn_output + self.attn.b_O

    return attn_output

  def compute_test_metrics(self):
    # compute MSE and accuracy on each batch and take average at the end
    mse = t.tensor(0.0, device=self.device)
    accuracy = t.tensor(0.0, device=self.device)
    batches = 0

    for batch in self.test_loader:
      inputs = t.tensor(batch[CaseDataset.INPUT_FIELD])
      expected_outputs = t.tensor(batch[CaseDataset.CORRECT_OUTPUT_FIELD])

      attn_output = self.run_attn(inputs)

      mse = t.nn.functional.mse_loss(expected_outputs, attn_output)
      accuracy = accuracy + t.isclose(expected_outputs, attn_output,
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