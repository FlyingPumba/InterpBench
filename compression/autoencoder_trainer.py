import numpy as np
import torch as t
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.benchmark_case import BenchmarkCase
from compression.autencoder import AutoEncoder
from compression.autoencoder_training_args import AutoEncoderTrainingArgs
from utils.hooked_tracr_transformer import HookedTracrTransformer


class AutoEncoderTrainer:
  ACTIVATIONS_FIELD = "activations"

  def __init__(self, case: BenchmarkCase,
               autoencoder: AutoEncoder,
               tl_model: HookedTracrTransformer,
               args: AutoEncoderTrainingArgs):
    super().__init__()
    self.case = case
    self.autoencoder = autoencoder
    self.tl_model = tl_model
    self.tl_model_n_layers = tl_model.cfg.n_layers

    self.tl_model.freeze_all_weights()
    self.device = autoencoder.device

    self.args = args
    self.use_wandb = self.args.ae_wandb_project is not None

    self.step = 0
    self.train_loss = np.nan
    self.test_metrics = {}

    self.setup_dataset(args)

    # calculate number of epochs and steps
    assert self.args.ae_epochs is not None or self.args.ae_steps is not None, "Must specify either epochs or steps."
    self.epochs = self.args.ae_epochs if self.args.ae_epochs is not None else int(self.args.ae_steps // len(self.train_loader)) + 1
    self.steps = self.args.ae_steps if self.args.ae_steps is not None else self.epochs * len(self.train_loader)

    self.optimizer = t.optim.AdamW(self.autoencoder.parameters(),
                                   lr=args.ae_lr_start,
                                   weight_decay=args.ae_weight_decay,
                                   betas=(args.ae_beta_1, args.ae_beta_2))

    # Learning rate scheduler with linear decay
    self.lr_warmup_steps = args.ae_lr_warmup_steps
    if self.lr_warmup_steps is None:
      # by default, half of total steps
      self.lr_warmup_steps = self.steps // 2

    lr_lambda = lambda step: max(args.ae_lr_end,
                                 args.ae_lr_start - (args.ae_lr_start - args.ae_lr_end) * (step / self.lr_warmup_steps))
    self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

    if self.use_wandb and self.args.ae_wandb_name is None:
      self.args.ae_wandb_name = f"case-{self.case.index_str}-autoencoder-{self.autoencoder.compression_size}"

  def setup_dataset(self, args: AutoEncoderTrainingArgs):
    """Prepare the dataset and split it into train and test sets."""
    tl_dataset = self.case.get_clean_data(count=args.ae_train_data_size)
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

    if args.ae_test_data_ratio is not None:
      # split the data into train and test sets
      test_data_size = int(len(tl_activations) * args.ae_test_data_ratio)
      train_data_size = len(tl_activations) - test_data_size
      train_data, test_data = tl_activations.split([train_data_size, test_data_size])
    else:
      # use all data for training, and the same data for testing
      train_data = tl_activations
      test_data = tl_activations

    self.train_loader = DataLoader(train_data, batch_size=args.ae_batch_size, shuffle=True)
    self.test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

  def train(self):
    """
    Trains the model, for `self.args.ae_epochs` epochs.
    """
    print(f'Starting run to train autoencoder: {self.autoencoder.input_size} -> {self.autoencoder.compression_size}.')

    if self.use_wandb:
      wandb.init(project=self.args.ae_wandb_project, name=self.args.ae_wandb_name, config=self.args)

    progress_bar = tqdm(total=len(self.train_loader) * self.epochs)
    for epoch in range(self.epochs):
      for i, batch in enumerate(self.train_loader):
        self.train_loss = self.training_step(batch)

        progress_bar.update()
        progress_bar.set_description(f"Epoch {epoch + 1}, train_loss: {self.train_loss:.3f}" +
                                     self.build_test_metrics_string())

      self.evaluate_test_metrics()

      if (self.args.ae_early_stop_test_accuracy is not None and
          self.test_metrics["test_accuracy"] >= self.args.ae_early_stop_test_accuracy):
        break

    if self.use_wandb:
      wandb.finish()

    return {**self.test_metrics, "train_loss": self.train_loss.item()}

  def training_step(self, inputs: Float[Tensor, "batch_size d_model"]) -> Float[Tensor, ""]:
    '''
    Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

    Remember that `batch` is a dictionary with the single key 'tokens'.
    '''
    self.optimizer.zero_grad()

    # Run the input on both compressed and original model
    autoencoder_output = self.autoencoder(inputs)

    # compute the loss
    loss = self.compute_loss(inputs, autoencoder_output)

    loss.backward()
    self.optimizer.step()
    self.lr_scheduler.step()

    self.step += 1

    return loss

  def compute_loss(
      self,
      autoencoder_input: Float[Tensor, "batch_size d_model"],
      autoencoder_output: Float[Tensor, "batch_size d_model"],
  ) -> Float[Tensor, "batch posn-1"]:
    loss = t.nn.functional.mse_loss(autoencoder_input, autoencoder_output)

    if self.use_wandb:
      wandb.log({"train_loss": loss}, step=self.step)

    return loss

  def evaluate_test_metrics(self):
    inputs = next(iter(self.test_loader))
    outputs = self.autoencoder(inputs)

    self.test_metrics["test_mse"] = t.nn.functional.mse_loss(inputs, outputs).item()

    if self.use_wandb:
      wandb.log(self.test_metrics, step=self.step)

  def build_test_metrics_string(self):
    if len(self.test_metrics.items()) == 0:
      return ""
    else:
      return ", " + ("".join([f"{k}: {v:.3f}, " for k, v in self.test_metrics.items()]))[:-2]
