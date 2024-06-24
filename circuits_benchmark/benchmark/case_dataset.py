from __future__ import annotations

from torch.utils.data import DataLoader, Dataset


class CaseDataset(Dataset):
  def get_inputs(self):
    raise NotImplementedError()

  def get_targets(self):
    raise NotImplementedError()

  @staticmethod
  def collate_fn(batch):
    raise NotImplementedError()

  def make_loader(
      self,
      batch_size: int | None = None,
      shuffle: bool | None = False,
  ) -> DataLoader:
    raise NotImplementedError()
