from datasets import Dataset


class CaseDataset(Dataset):
  def get_inputs(self):
    raise NotImplementedError()

  def get_targets(self):
    raise NotImplementedError()