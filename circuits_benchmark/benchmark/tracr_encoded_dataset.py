from torch import Tensor
from torch.utils.data import Dataset


class TracrEncodedDataset(Dataset):
  """Same as TracrDataset, but with encoded inputs and outputs (i.e., tensors instead of numpy arrays)."""

  def __init__(self,
               inputs: Tensor,
               expected_outputs: Tensor):
    self.inputs = inputs
    self.expected_outputs = expected_outputs

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    return self.inputs[idx], self.expected_outputs[idx]
