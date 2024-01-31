from dataclasses import dataclass
from typing import Optional


@dataclass
class AutoEncoderTrainingArgs():
  """Dataclass for autoencoder training arguments. All fields are the same as CompressionTrainingArgs, but their name is
   prefixed with "ae_".
   """
  # Wandb config
  ae_wandb_project: Optional[str] = None
  ae_wandb_name: Optional[str] = None

  # data management
  ae_batch_size: Optional[int] = 256
  ae_train_data_size: Optional[int] = None  # use all data available
  ae_test_data_ratio: Optional[float] = None  # same as train data

  # training time and early stopping
  ae_epochs: Optional[int] = None
  ae_steps: Optional[int] = 3e5
  ae_early_stop_test_accuracy: Optional[float] = None

  # AdamW optimizer config
  ae_weight_decay: Optional[float] = 0.1
  ae_beta_1: Optional[float] = 0.9
  ae_beta_2: Optional[float] = 0.99

  # learning rate config
  ae_lr_warmup_steps: Optional[float] = None  # by default, first half of total steps
  ae_lr_start: Optional[float] = 1e-3
  ae_lr_end: Optional[float] = 1e-6

  # test metrics config
  ae_test_accuracy_atol: Optional[float] = 1e-2

  # compression layers
  ae_n_layers: Optional[int] = 2

  # load autoencoder from file if available
  ae_load_from_file: Optional[str] = None
  ae_save_to_file: Optional[str] = None