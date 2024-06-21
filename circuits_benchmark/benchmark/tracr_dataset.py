import numpy as np
import torch as t
from torch.utils.data import Dataset

from circuits_benchmark.benchmark.tracr_encoded_dataset import TracrEncodedDataset
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer


class TracrDataset(Dataset):
  def __init__(self,
               inputs: np.ndarray,
               expected_outputs: np.ndarray,
               hl_model: HookedTracrTransformer | None = None):
    self.inputs = inputs
    self.expected_outputs = expected_outputs
    assert inputs.shape == expected_outputs.shape
    self.hl_model = hl_model

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    return self.inputs[idx], self.expected_outputs[idx]

  def get_inputs(self):
    return self.inputs

  def get_expected_outputs(self):
    return self.expected_outputs

  def get_encoded_dataset(
      self,
      device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
  ):
      encoded_inputs = self.hl_model.map_tracr_input_to_tl_input(self.inputs)
      with t.no_grad():
          encoded_outputs = self.hl_model(encoded_inputs)

      return TracrEncodedDataset(encoded_inputs.to(device), encoded_outputs.to(device))
