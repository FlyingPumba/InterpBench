import importlib
import os.path
from typing import Set, Optional

import numpy as np
from networkx import DiGraph
from torch import Tensor

from benchmark.case_dataset import CaseDataset
from tracr.compiler.assemble import AssembledTransformerModel
from tracr.rasp import rasp
from utils.cloudpickle import load_from_pickle, dump_to_pickle
from utils.detect_project_root import detect_project_root
from utils.hooked_tracr_transformer import HookedTracrTransformer, HookedTracrTransformerBatchInput


class BenchmarkCase(object):

  def __init__(self):
    self.case_file_absolute_path = os.path.join(detect_project_root(), self.get_relative_path_from_root())
    self.data_generation_seed = 42

  @staticmethod
  def get_instance_for_file_path(file_path_from_root: str):
    """Returns an instance of the benchmark case for the corresponding given file path."""
    # load module for the file
    # The module name is built by replacing "/" with "." and removing the ".py" extension from the file path
    module_name = file_path_from_root.replace("/", ".")[:-3]
    module = importlib.import_module(module_name)

    # Create class for case dinamically
    index_str = file_path_from_root.split("/")[1].split("-")[1]
    class_name = f"Case{index_str}"
    case_instance = getattr(module, class_name)(file_path_from_root)

    return case_instance

  def get_program(self) -> rasp.SOp:
    """Returns the RASP program to be compiled by Tracr."""
    raise NotImplementedError()

  def get_vocab(self) -> Set:
    """Returns the vocabulary to be used by Tracr."""
    raise NotImplementedError()

  def get_clean_data(self, count: Optional[int] = 10) -> CaseDataset:
    """Returns the clean data for the benchmark case."""
    seq_len = self.get_max_seq_len()
    input_data: HookedTracrTransformerBatchInput = []
    output_data: HookedTracrTransformerBatchInput = []

    # set numpy seed and sort vocab to ensure reproducibility
    np.random.seed(self.data_generation_seed)
    vals = sorted(list(self.get_vocab()))

    # If count is None, we will produce all possible sequences for this vocab and sequence length
    produce_all = False
    if count is None:
      count = len(vals) ** (seq_len - 1)
      produce_all = True

    for index in range(count):
      if produce_all:
        # we want to produce all possible sequences, so we convert the index to base len(vals) and then convert each
        # digit to the corresponding value in vals
        sample = []
        base = len(vals)
        num = index
        while num:
          sample.append(vals[num % base])
          num //= base

        if len(sample) < seq_len - 1:
          # extend with the first value in vocab to fill the sequence
          sample.extend([vals[0]] * (seq_len - 1 - len(sample)))

        # reverse the list to produce the sequence in the correct order
        sample = sample[::-1]
      else:
        sample = np.random.choice(vals, size=seq_len - 1).tolist()  # sample with replacement

      output = self.get_program()(sample)

      input_data.append(["BOS"] + sample)
      output_data.append(["BOS"] + output)

    return CaseDataset(input_data, output_data)

  def get_validation_metric(self, metric_name: str, tl_model: HookedTracrTransformer) -> Tensor:
    """Returns the validation metric for the benchmark case."""
    raise NotImplementedError()

  def get_corrupted_data(self, count: int = 10) -> CaseDataset:
    """Returns the corrupted data for the benchmark case.
    Default implementation: re-generate clean data with a different seed."""
    self.data_generation_seed = self.data_generation_seed + 1
    dataset = self.get_clean_data(count=count)
    self.data_generation_seed = 42
    return dataset

  def get_max_seq_len(self) -> int:
    """Returns the maximum sequence length for the benchmark case.
    Default implementation: 10."""
    return 10

  def get_index(self) -> str:
    class_name = self.__class__.__name__  # Looks like "CaseN"
    return class_name[4:]

  def __str__(self):
    return self.case_file_absolute_path

  def get_tracr_model_pickle_path(self) -> str:
    return self.case_file_absolute_path.replace(".py", "_tracr_model.pkl")

  def get_tracr_graph_pickle_path(self) -> str:
    return self.case_file_absolute_path.replace(".py", "_tracr_graph.pkl")

  def get_tl_model_pickle_path(self) -> str:
    return self.case_file_absolute_path.replace(".py", "_tl_model.pkl")

  def load_tracr_model(self) -> AssembledTransformerModel | None:
    """Loads the tracr model from disk, if it exists."""
    return load_from_pickle(self.get_tracr_model_pickle_path())

  def load_tracr_graph(self) -> DiGraph | None:
    """Loads the tracr graph from disk, if it exists."""
    return load_from_pickle(self.get_tracr_graph_pickle_path())

  def load_tl_model(self) -> HookedTracrTransformer | None:
    """Loads the transformer_lens model from disk, if it exists."""
    return load_from_pickle(self.get_tl_model_pickle_path())

  def dump_tracr_model(self, tracr_model: AssembledTransformerModel) -> None:
    """Dumps the tracr model to disk."""
    dump_to_pickle(self.get_tracr_model_pickle_path(), tracr_model)

  def dump_tracr_graph(self, tracr_graph: DiGraph) -> None:
    """Dumps the tracr graph to disk."""
    dump_to_pickle(self.get_tracr_graph_pickle_path(), tracr_graph)

  def dump_tl_model(self, tl_model: HookedTracrTransformer) -> None:
    """Dumps the transformer_lens model to disk."""
    dump_to_pickle(self.get_tl_model_pickle_path(), tl_model)

  def get_relative_path_from_root(self) -> str:
    return f"benchmark/cases/case_{self.get_index()}.py"