import importlib
import os
from typing import Set, Tuple

import numpy as np
from cloudpickle import cloudpickle
from networkx import DiGraph
from torch import Tensor

from tracr.compiler.assemble import AssembledTransformerModel
from tracr.rasp import rasp
from utils.hooked_tracr_transformer import HookedTracrTransformer, HookedTracrTransformerBatchInput
from utils.relativize_path import relativize_path


class BenchmarkCase(object):
  def __init__(self, file_path_from_root: str):
    self.file_path_from_root = file_path_from_root
    self.index_str = file_path_from_root.split("/")[1].split("-")[1]

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

  def get_clean_data(self, count: int = 10) -> Tuple[HookedTracrTransformerBatchInput, HookedTracrTransformerBatchInput]:
    """Returns a tuple of (input, expected_output) for the benchmark case."""
    raise NotImplementedError()

  def get_validation_metric(self, tl_model: HookedTracrTransformer) -> Tensor:
    """Returns the validation metric for the benchmark case."""
    raise NotImplementedError()

  def get_corrupted_data(self, count: int = 10) -> HookedTracrTransformerBatchInput:
    """Returns the corrupted data for the benchmark case.
    Default implementation: random permutation of clean data."""
    clean_data, _ = self.get_clean_data(count=count)
    patch_data_indices = np.random.permutation(len(clean_data))
    corrupted_data = np.array(clean_data)[patch_data_indices].tolist()
    return corrupted_data

  def get_max_seq_len(self) -> int:
    """Returns the maximum sequence length for the benchmark case.
    Default implementation: 10."""
    return 10

  def get_file_path_from_root(self) -> str:
    return self.file_path_from_root

  def get_benchmark_index(self) -> str:
    return self.index_str

  def __str__(self):
    return self.file_path_from_root

  def get_tracr_model_path_from_root(self) -> str:
    return self.file_path_from_root.replace("rasp.py", "tracr_model.pkl")

  def get_tracr_graph_path_from_root(self) -> str:
    return self.file_path_from_root.replace("rasp.py", "tracr_graph.pkl")

  def get_tl_model_path_from_root(self) -> str:
    return self.file_path_from_root.replace("rasp.py", "tl_model.pkl")

  def load_tracr_model(self) -> AssembledTransformerModel | None:
    """Loads the tracr model from disk, if it exists."""
    tracr_model_output_path = relativize_path(self.get_tracr_model_path_from_root())
    return self.load_from_pickle(tracr_model_output_path)

  def load_tracr_graph(self) -> DiGraph | None:
    """Loads the tracr graph from disk, if it exists."""
    tracr_graph_output_path = relativize_path(self.get_tracr_graph_path_from_root())
    return self.load_from_pickle(tracr_graph_output_path)

  def load_tl_model(self) -> HookedTracrTransformer | None:
    """Loads the transformer_lens model from disk, if it exists."""
    tl_model_output_path = relativize_path(self.get_tl_model_path_from_root())
    return self.load_from_pickle(tl_model_output_path)

  def dump_tracr_model(self, tracr_model: AssembledTransformerModel) -> None:
    """Dumps the tracr model to disk."""
    tracr_model_output_path = relativize_path(self.get_tracr_model_path_from_root())
    self.dump_to_pickle(tracr_model_output_path, tracr_model)

  def dump_tracr_graph(self, tracr_graph: DiGraph) -> None:
    """Dumps the tracr graph to disk."""
    tracr_graph_output_path = relativize_path(self.get_tracr_graph_path_from_root())
    self.dump_to_pickle(tracr_graph_output_path, tracr_graph)

  def dump_tl_model(self, tl_model: HookedTracrTransformer) -> None:
    """Dumps the transformer_lens model to disk."""
    tl_model_output_path = relativize_path(self.get_tl_model_path_from_root())
    self.dump_to_pickle(tl_model_output_path, tl_model)

  def load_from_pickle(self, path) -> object | None:
    if os.path.exists(path):
      with open(path, "rb") as f:
        return cloudpickle.load(f)
    else:
      return None

  def dump_to_pickle(self, path, obj) -> None:
    with open(path, "wb") as f:
      cloudpickle.dump(obj, f)


