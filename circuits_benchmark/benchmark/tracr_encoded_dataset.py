import torch as t
from torch import Tensor
from torch.utils.data import DataLoader

from circuits_benchmark.benchmark.case_dataset import CaseDataset


class TracrEncodedDataset(CaseDataset):
  """Same as TracrDataset, but with encoded inputs and outputs (i.e., tensors instead of numpy arrays)."""

  def __init__(self,
               inputs: Tensor,
               targets: Tensor):
    self.inputs = inputs
    self.targets = targets

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
    inputs = t.stack([x[0] for x in batch])
    targets = t.stack([x[1] for x in batch])
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
