import numpy as np
import torch as t

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

  def get_encoded_dataset(
      self,
      device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
  ):
      encoded_inputs = self.hl_model.map_tracr_input_to_tl_input(self.inputs)
      with t.no_grad():
          encoded_outputs = self.hl_model(encoded_inputs)

      return TracrEncodedDataset(encoded_inputs.to(device), encoded_outputs.to(device))
