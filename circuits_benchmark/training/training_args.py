from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingArgs():
  verbose: Optional[bool] = False
  # Wandb config
  wandb_project: Optional[str] = None
  wandb_name: Optional[str] = None

  # data management
  batch_size: Optional[int] = 256
  min_train_samples: Optional[int] = 20_000
  max_train_samples: Optional[int] = 120_000
  test_data_ratio: Optional[float] = 0.2  # same as train data

  # training time and early stopping
  epochs: Optional[int] = None
  steps: Optional[int] = None
  early_stop_threshold: Optional[float] = None

  # AdamW optimizer config
  weight_decay: Optional[float] = 0
  beta_1: Optional[float] = 0.9
  beta_2: Optional[float] = 0.95
  gradient_clip: Optional[float] = 1

  # lr scheduler config
  lr_start: Optional[float] = 1e-2
  lr_factor: Optional[float] = 0.75
  lr_patience: Optional[int] = 10
  lr_threshold: Optional[float] = 1e-4

  # test metrics config
  test_accuracy_atol: Optional[float] = 5e-2

  # resample ablation loss config
  resample_ablation_test_loss: Optional[bool] = False
  resample_ablation_loss_epochs_gap: Optional[int] = 50
  resample_ablation_max_interventions: Optional[int] = 10
  resample_ablation_max_components: Optional[int] = 1
  resample_ablation_batch_size: Optional[int] = 20000
  resample_ablation_loss_weight: Optional[float] = 1
