from typing import Dict, List, Any

import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader

from training.training_args import TrainingArgs
from utils.hooked_tracr_transformer import HookedTracrTransformerBatchInput


class CaseDataset(Dataset):
  INPUT_FIELD = "input"
  CORRECT_OUTPUT_FIELD = "correct_output"

  def __init__(self, input_data: HookedTracrTransformerBatchInput,
               output_data: HookedTracrTransformerBatchInput):
    self.dataframe = pd.DataFrame({
      self.INPUT_FIELD: input_data,
      self.CORRECT_OUTPUT_FIELD: output_data
    })

  def get_inputs(self):
    return self.dataframe[self.INPUT_FIELD]

  def get_correct_outputs(self):
    return self.dataframe[self.CORRECT_OUTPUT_FIELD]

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    """Returns the requested samples."""
    batch = self.dataframe.iloc[idx]
    batch.reset_index(inplace=True, drop=True)
    return batch

  def train_test_split(self, args: TrainingArgs, shuffle_before_split: bool = True) -> (DataLoader, DataLoader):
    test_data_ratio = args.test_data_ratio

    # by default, we use the same data for training and testing
    train_data: Dataset = self
    test_data: Dataset = self

    if shuffle_before_split:
      self.dataframe = self.dataframe.sample(frac=1, axis=0)
      self.dataframe.reset_index(inplace=True, drop=True)

    if args.test_data_ratio is not None:
      test_dataframe = self.dataframe.sample(frac=test_data_ratio, axis=0)
      test_dataframe.reset_index(inplace=True, drop=True)
      test_data = CaseDataset(list(test_dataframe[self.INPUT_FIELD]),
                              list(test_dataframe[self.CORRECT_OUTPUT_FIELD]))

      # get everything but the test sample
      train_dataframe = self.dataframe.drop(index=test_dataframe.index)
      train_dataframe.reset_index(inplace=True, drop=True)
      train_data = CaseDataset(list(train_dataframe[self.INPUT_FIELD]),
                               list(train_dataframe[self.CORRECT_OUTPUT_FIELD]))

    # prepare data loaders
    def custom_collate(items: List[Dict[str, List[Any]]]) -> dict[str, HookedTracrTransformerBatchInput]:
      return {self.INPUT_FIELD: [item[self.INPUT_FIELD] for item in items],
              self.CORRECT_OUTPUT_FIELD: [item[self.CORRECT_OUTPUT_FIELD] for item in items]}

    batch_size = args.batch_size if args.batch_size is not None else len(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=custom_collate)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False,
                             collate_fn=custom_collate)

    return train_loader, test_loader
