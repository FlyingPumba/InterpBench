from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingArgs():
  # Wandb config
  wandb_project: Optional[str] = None
  wandb_name: Optional[str] = None

  # data management
  batch_size: Optional[int] = 256
  train_data_size: Optional[int] = None  # use all data available
  test_data_ratio: Optional[float] = None  # same as train data

  # training time and early stopping
  epochs: Optional[int] = None
  steps: Optional[int] = 3e5
  early_stop_test_accuracy: Optional[float] = None

  # AdamW optimizer config
  weight_decay: Optional[float] = 0.1
  beta_1: Optional[float] = 0.9
  beta_2: Optional[float] = 0.99

  # learning rate config
  lr_warmup_steps: Optional[float] = None  # by default, first half of total steps
  lr_start: Optional[float] = 1e-3
  lr_end: Optional[float] = 1e-6

  # test metrics config
  test_accuracy_atol: Optional[float] = 1e-2