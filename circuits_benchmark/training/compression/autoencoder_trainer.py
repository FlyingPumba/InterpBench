import torch as t
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.training.compression.autencoder import AutoEncoder
from circuits_benchmark.training.compression.compression_train_loss_level import CompressionTrainLossLevel
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
               train_loss_level: CompressionTrainLossLevel = "layer",
               output_dir: str | None = None):
    self.autoencoder = autoencoder
    self.tl_model = tl_model
    self.tl_model_n_layers = tl_model.cfg.n_layers
    self.tl_model.freeze_all_weights()
    self.train_loss_level = train_loss_level
    super().__init__(case, list(autoencoder.parameters()), args, output_dir=output_dir)

  def setup_dataset(self):
    tl_dataset = self.case.get_clean_data(count=self.args.train_data_size)
    tl_inputs = tl_dataset.get_inputs()
    _, tl_cache = self.tl_model.run_with_cache(tl_inputs)

    named_data = {}

    if self.train_loss_level == "layer":
      # collect the residual stream activations from all layers
      for layer in range(self.tl_model_n_layers):
        named_data[f"layer_{layer}_resid_pre"] = tl_cache["resid_pre", layer]
      named_data[f"layer_{self.tl_model_n_layers-1}_resid_post"] = tl_cache["resid_post", self.tl_model_n_layers-1]

    elif self.train_loss_level == "component":
      # collect the output of the attention and mlp components from all layers
      for layer in range(self.tl_model_n_layers):
        named_data[f"layer_{layer}_attn_out"] = tl_cache["attn_out", layer]
        named_data[f"layer_{layer}_mlp_out"] = tl_cache["mlp_out", layer]

      # collect the embeddings, but repeat the data self.tl_model_n_layers times
      named_data["embed"] = tl_cache["hook_embed"].repeat(self.tl_model_n_layers, 1, 1)
      named_data["pos_embed"] = tl_cache["hook_pos_embed"].repeat(self.tl_model_n_layers, 1, 1)
    else:
      raise ValueError(f"Invalid train_loss_level: {self.train_loss_level}")

    # Shape of tensors is [activations_len, seq_len, d_model]. We will convert to [activations_len, d_model] to
    # treat the residual stream for each sequence position as a separate sample.
    # The final length of all activations together is train_data_size*(n_layers + 1)*seq_len
    for name, data in named_data.items():
      named_data[name]: Float[Tensor, "activations_len, seq_len, d_model"] = (
        data.transpose(0, 1).reshape(-1, data.shape[-1]))

    # split the data into train and test sets
    self.named_test_data = {}
    self.named_train_data = {}
    if self.args.test_data_ratio is not None:
      # split the data into train and test sets
      for name, data in named_data.items():
        data_size = data.shape[0]
        test_data_size = int(data_size * self.args.test_data_ratio)
        train_data_size = data_size - test_data_size
        self.named_train_data[name], self.named_test_data[name] = t.split(data, [train_data_size, test_data_size])
    else:
      # use all data for training, and the same data for testing
      for name, data in named_data.items():
        self.named_train_data[name] = data.clone()
        self.named_test_data[name] = data.clone()

    # shuffle the activations in train and test data
    for name, data in self.named_train_data.items():
      self.named_train_data[name] = data[t.randperm(len(data))]
    for name, data in self.named_test_data.items():
      self.named_test_data[name] = data[t.randperm(len(data))]

    # collect all the train data into a single tensor
    train_data = t.cat(list(self.named_train_data.values()), dim=0)
    self.batch_size = self.args.batch_size if self.args.batch_size is not None else len(train_data)
    self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

  def compute_train_loss(self, batch: Float[Tensor, "batch_size d_model"]) -> Float[Tensor, "batch posn-1"]:
    output = self.autoencoder(batch)

    loss = t.nn.functional.mse_loss(batch, output)

    if self.use_wandb:
      wandb.log({"train_loss": loss}, step=self.step)

    return loss

  def compute_test_metrics(self):
    # compute MSE and accuracy on each batch and take average at the end
    test_mse = t.tensor(0.0, device=self.device)
    test_accuracy = t.tensor(0.0, device=self.device)
    test_batches = 0

    for name, test_data in self.named_test_data.items():
      self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

      named_mse = t.tensor(0.0, device=self.device)
      named_accuracy = t.tensor(0.0, device=self.device)
      batches = 0

      for inputs in self.test_loader:
        outputs = self.autoencoder(inputs)

        named_mse = named_mse + t.nn.functional.mse_loss(inputs, outputs)
        named_accuracy = named_accuracy + t.isclose(inputs, outputs, atol=self.args.test_accuracy_atol).float().mean()

        batches = batches + 1
        test_batches = test_batches + 1

      test_mse = test_mse + named_mse
      test_accuracy = test_accuracy + named_accuracy

      named_mse = named_mse / batches
      self.test_metrics[f"test_{name}_mse"] = named_mse.item()

    self.test_metrics["test_mse"] = (test_mse / test_batches).item()
    self.test_metrics["test_accuracy"] = (test_accuracy / test_batches).item()

    if self.use_wandb:
      wandb.log(self.test_metrics, step=self.step)

  def build_wandb_name(self):
    return f"case-{self.case.get_index()}-autoencoder-{self.autoencoder.compression_size}"

  def get_wandb_tags(self):
    tags = super().get_wandb_tags()
    tags.append("autoencoder-trainer")
    return tags

  def get_wandb_config(self):
    cfg = super().get_wandb_config()
    cfg.update({
      "train_loss_level": self.train_loss_level,
      "ae_input_size": self.autoencoder.input_size,
      "ae_compression_size": self.autoencoder.compression_size,
      "ae_layers": self.autoencoder.n_layers,
      "ae_first_hidden_layer_shape": self.autoencoder.first_hidden_layer_shape,
      "ae_use_bias": self.autoencoder.use_bias,
    })
    return cfg

  def save_artifacts(self):
    prefix = f"case-{self.case.get_index()}-resid-{self.autoencoder.compression_size}"
    self.autoencoder.save(self.output_dir, prefix, self.wandb_run)