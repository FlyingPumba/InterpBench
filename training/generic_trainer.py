from typing import List

import numpy as np
import torch as t
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.nn import Parameter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.benchmark_case import BenchmarkCase
from training.training_args import TrainingArgs


class GenericTrainer():

  def __init__(self,
               case: BenchmarkCase,
               parameters: List[Parameter],
               training_args: TrainingArgs):
    self.case = case
    self.parameters = parameters
    self.args = training_args

    self.use_wandb = self.args.wandb_project is not None

    self.step = 0
    self.train_loss = np.nan
    self.test_metrics = {}

    self.train_loader: DataLoader = None
    self.test_loader: DataLoader = None
    self.setup_dataset()

    # calculate number of epochs and steps
    assert self.args.epochs is not None or self.args.steps is not None, "Must specify either epochs or steps."
    self.epochs = self.args.epochs if self.args.epochs is not None else int(self.args.steps // len(self.train_loader)) + 1
    self.steps = self.args.steps if self.args.steps is not None else self.epochs * len(self.train_loader)

    # assert at least one parameter
    assert len(self.parameters) > 0, "No parameters to optimize."

    # get the device from parameters
    self.device = self.parameters[0].device

    self.optimizer = t.optim.AdamW(self.parameters,
                                   lr=self.args.lr_start,
                                   weight_decay=self.args.weight_decay,
                                   betas=(self.args.beta_1, self.args.beta_2))

    # Learning rate scheduler with linear decay
    self.lr_warmup_steps = self.args.lr_warmup_steps
    if self.lr_warmup_steps is None:
      # by default, half of total steps
      self.lr_warmup_steps = self.steps // 2

    lr_lambda = lambda step: max(self.args.lr_end,
                                 self.args.lr_start - (self.args.lr_start - self.args.lr_end) * (step / self.lr_warmup_steps))
    self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

    if self.use_wandb and self.args.wandb_name is None:
      self.args.wandb_name = self.build_wandb_name()

  def setup_dataset(self):
    """Prepare the dataset and split it into train and test sets."""
    raise NotImplementedError

  def train(self):
    """
    Trains the model, for `self.args.epochs` epochs.
    """
    if self.use_wandb:
      wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)

    progress_bar = tqdm(total=len(self.train_loader) * self.epochs)
    for epoch in range(self.epochs):
      for i, batch in enumerate(self.train_loader):
        self.train_loss = self.training_step(batch)

        progress_bar.update()
        progress_bar.set_description(f"Epoch {epoch + 1}, train_loss: {self.train_loss:.3f}" +
                                     self.build_test_metrics_string())

      self.compute_test_metrics()

      if (self.args.early_stop_test_accuracy is not None and
          self.test_metrics["test_accuracy"] >= self.args.early_stop_test_accuracy):
        break

    if self.use_wandb:
      wandb.finish()

    return {**self.test_metrics, "train_loss": self.train_loss.item()}

  def training_step(self, inputs) -> Float[Tensor, ""]:
    """Calculates the loss on batched inputs, performs a gradient update step, and logs the loss."""
    self.optimizer.zero_grad()

    loss = self.compute_train_loss(inputs)

    self.update_params(loss)

    self.step += 1

    return loss

  def update_params(self, loss: Float[Tensor, ""]):
    """Performs a gradient update step."""
    loss.backward()
    self.optimizer.step()
    self.lr_scheduler.step()

  def compute_train_loss(self, inputs) -> Float[Tensor, ""]:
    raise NotImplementedError

  def compute_test_metrics(self):
    raise NotImplementedError

  def build_test_metrics_string(self):
    if len(self.test_metrics.items()) == 0:
      return ""
    else:
      return ", " + ("".join([f"{k}: {v:.3f}, " for k, v in self.test_metrics.items()]))[:-2]

  def build_wandb_name(self):
    return f"case-{self.case.index_str}-generic-training"