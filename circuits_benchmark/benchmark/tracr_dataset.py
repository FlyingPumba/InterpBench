from __future__ import annotations

import numpy as np
import torch as t
from torch.utils.data import DataLoader

from circuits_benchmark.benchmark.case_dataset import CaseDataset
from circuits_benchmark.benchmark.tracr_encoded_dataset import TracrEncodedDataset
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer


class TracrDataset(CaseDataset):
  def __init__(self,
               inputs: np.ndarray,
               targets: np.ndarray,
               hl_model: HookedTracrTransformer | None = None):
    self.inputs = inputs
    self.targets = targets
    assert inputs.shape == targets.shape
    self.hl_model = hl_model

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    return self.inputs[idx], self.targets[idx]

  def get_inputs(self):
    return self.inputs

  def get_targets(self):
    return self.targets

  @staticmethod
  def collate_fn(batch):
    inputs = [x[0] for x in batch]
    targets = [x[1] for x in batch]
    return inputs, targets

  def make_loader(
      self,
      batch_size: int | None = None,
      shuffle: bool | None = False,
  ) -> DataLoader:
    return DataLoader(
      self,
      batch_size=batch_size,
      shuffle=shuffle,
      collate_fn=lambda x: self.collate_fn(x),
    )

  def get_encoded_dataset(self):
    encoded_inputs = self.hl_model.map_tracr_input_to_tl_input(self.inputs)
    with t.no_grad():
      encoded_outputs = self.hl_model(encoded_inputs)

    return TracrEncodedDataset(encoded_inputs, encoded_outputs)
